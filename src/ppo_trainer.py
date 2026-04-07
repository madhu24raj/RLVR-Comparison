"""
PPO trainer for RLVR on GSM8K using TRL.

Wraps TRL's PPOTrainer with our reward function, critic module,
and experiment logging. Designed for the E2.7 head-to-head comparison
and E2.8 critic architecture sweep.

PPO overview:
    1. Generate completions for a batch of prompts (rollout)
    2. Score completions with the reward function (binary: correct/incorrect)
    3. Estimate advantages using the critic: A = R - V(s)
    4. Update the policy to increase probability of high-advantage completions
    5. Update the critic to better predict future rewards
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset

from src.data import get_experiment_subset, format_prompt
from src.rewards import gsm8k_reward, batch_reward
from src.critic import build_critic, CRITIC_CONFIGS
from eval.metrics import ExperimentLogger, accuracy


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B"):
    """Load the base model with a value head and tokenizer.

    The value head is TRL's built-in mechanism for PPO — it attaches
    a linear layer on top of the base model to estimate values.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        model: AutoModelForCausalLMWithValueHead
        tokenizer: AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model, tokenizer


def prepare_dataset(n_prompts: int = 100, seed: int = 42) -> Dataset:
    """Prepare the training dataset in the format PPOTrainer expects.

    Each row needs a 'query' field containing the tokenized prompt.
    Ground truths are stored alongside for reward computation.
    """
    train_data, _ = get_experiment_subset(n=n_prompts, seed=seed)

    formatted = []
    for example in train_data:
        prompt = format_prompt(example["question"])
        formatted.append({
            "query": prompt,
            "ground_truth": example["ground_truth"],
        })

    return Dataset.from_list(formatted)


def run_ppo(
    model_name: str = "Qwen/Qwen2.5-7B",
    critic_config: str = "medium",
    n_prompts: int = 100,
    n_steps: int = 200,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    max_new_tokens: int = 512,
    learning_rate: float = 1e-5,
    seed: int = 42,
    experiment_name: str = None,
):
    """Run PPO training on GSM8K.

    Args:
        model_name: Base model to fine-tune
        critic_config: Critic size ('none', 'small', 'medium', 'large')
        n_prompts: Number of training prompts
        n_steps: Total PPO update steps
        batch_size: Prompts per PPO step
        mini_batch_size: Mini-batch size for PPO updates
        max_new_tokens: Max tokens to generate per completion
        learning_rate: Learning rate for both policy and value head
        seed: Random seed
        experiment_name: Name for logging (auto-generated if None)
    """
    if experiment_name is None:
        experiment_name = f"ppo_critic_{critic_config}_seed{seed}"

    print(f"=== PPO Training ===")
    print(f"Model: {model_name}")
    print(f"Critic: {critic_config} {CRITIC_CONFIGS[critic_config]}")
    print(f"Prompts: {n_prompts}, Steps: {n_steps}")
    print(f"Seed: {seed}")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Configure PPO
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        seed=seed,
    )

    # Prepare dataset
    dataset = prepare_dataset(n_prompts=n_prompts, seed=seed)

    # Build PPO trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Setup logging
    logger = ExperimentLogger(experiment_name)

    # Training loop
    print("Starting training...\n")
    for step in range(n_steps):
        # Sample a batch of prompts
        batch = dataset.shuffle(seed=seed + step).select(range(min(batch_size, len(dataset))))

        # Tokenize queries
        query_tensors = [
            tokenizer.encode(q, return_tensors="pt").squeeze()
            for q in batch["query"]
        ]

        # Generate completions
        response_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode completions
        completions = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        # Compute rewards
        rewards = batch_reward(completions, batch["ground_truth"])
        reward_tensors = [torch.tensor(r) for r in rewards]

        # PPO update step
        stats = trainer.step(query_tensors, response_tensors, reward_tensors)

        # Log metrics
        step_accuracy = accuracy(rewards)
        logger.log_step(
            step=step,
            accuracy=step_accuracy,
            reward_mean=sum(rewards) / len(rewards),
            reward_var=float(torch.var(torch.tensor(rewards))) if len(rewards) > 1 else 0.0,
            policy_loss=stats.get("ppo/loss/policy", 0.0),
            value_loss=stats.get("ppo/loss/value", 0.0),
            kl_divergence=stats.get("ppo/mean_kl", 0.0),
        )

        if (step + 1) % 10 == 0:
            print(
                f"Step {step+1}/{n_steps} | "
                f"Acc: {step_accuracy:.1%} | "
                f"Reward: {sum(rewards)/len(rewards):.3f} | "
                f"KL: {stats.get('ppo/mean_kl', 0):.4f}"
            )

    # Save results
    logger.save()
    print(f"\nTraining complete. Results saved to results/{experiment_name}.json")

    # Save the fine-tuned model
    model.save_pretrained(f"models/{experiment_name}")
    tokenizer.save_pretrained(f"models/{experiment_name}")
    print(f"Model saved to models/{experiment_name}")

    return logger


def evaluate_on_test(
    model_path: str,
    model_name: str = "Qwen/Qwen2.5-7B",
    max_new_tokens: int = 512,
    seed: int = 42,
) -> float:
    """Evaluate a fine-tuned model on the full GSM8K test set.

    Args:
        model_path: Path to the saved fine-tuned model
        model_name: Base model name (for tokenizer)
        max_new_tokens: Max generation length
        seed: Random seed

    Returns:
        Test accuracy as a float
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    _, test_data = get_experiment_subset(seed=seed)

    correct = 0
    total = 0

    for example in test_data:
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for evaluation
            )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reward = gsm8k_reward(completion, example["ground_truth"])
        correct += reward
        total += 1

    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.1%} ({int(correct)}/{total})")
    return test_acc


if __name__ == "__main__":
    # Dry run with small config for testing
    print("PPO trainer module loaded successfully.")
    print(f"Available critic configs: {list(CRITIC_CONFIGS.keys())}")
    print("\nTo run training:")
    print("  from src.ppo_trainer import run_ppo")
    print("  run_ppo(critic_config='medium', n_steps=100, seed=42)")
