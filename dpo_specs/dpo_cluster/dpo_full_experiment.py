"""
Full DPO Experiment Script
Covers E2.7 (head-to-head metrics) and E2.9 (label regime comparison)
across 3 seeds. Saves all results to JSON for paper reporting.
"""

import torch
import random
import json
import os
import re
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
MODEL_ID       = "meta-llama/Meta-Llama-3-8B"
OUTPUT_ROOT    = "/weka/scratch/rarora8/madhu/dpo_experiment/results"
SEEDS          = [42, 123, 7]
MAX_STEPS      = 200
TRAIN_SAMPLES  = 2000
EVAL_SAMPLES   = 200
MC_ROLLOUTS    = 50   # Monte Carlo rollouts for advantage estimation error (keep low for speed)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_answer(text):
    """Extract numeric answer after #### in GSM8K."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def create_synthetic_dpo_pairs(dataset_split, num_samples, mode="full", noise_rate=0.0, seed=42):
    """
    Build DPO preference pairs.
    mode:
      'full'   — all correct completions paired with random incorrect ones
      'sparse' — only 10% of examples labeled (rest dropped)
      'noisy'  — 10% of correct labels flipped to incorrect
    """
    random.seed(seed)
    dpo_data = {"prompt": [], "chosen": [], "rejected": []}
    samples = list(dataset_split)[:num_samples]

    if mode == "sparse":
        samples = random.sample(samples, max(1, int(len(samples) * 0.1)))

    for i, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        correct = item['answer']
        incorrect = random.choice([x for j, x in enumerate(samples) if j != i])['answer']

        # noisy: flip chosen/rejected for noise_rate fraction
        if mode == "noisy" and random.random() < noise_rate:
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(incorrect)   # flipped!
            dpo_data["rejected"].append(correct)
        else:
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(correct)
            dpo_data["rejected"].append(incorrect)

    return Dataset.from_dict(dpo_data)


def load_model_and_tokenizer():
    """Load quantized Llama-3-8B with LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def train_dpo(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """Run DPO training, return trainer with logged metrics."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=MAX_STEPS,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_length=512,
    )
    trainer.train()
    return trainer


def evaluate_accuracy(model, tokenizer, dataset_split, num_samples=200, seed=42):
    """
    Generate answers for GSM8K questions and compute exact-match accuracy.
    """
    random.seed(seed)
    samples = list(dataset_split)[:num_samples]
    model.eval()
    correct = 0
    reward_variances = []

    with torch.no_grad():
        for item in samples:
            prompt = f"Question: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            pred = extract_answer(generated)
            gold = extract_answer(item['answer'])

            if pred is not None and gold is not None and pred == gold:
                correct += 1

    accuracy = correct / len(samples)
    return accuracy


def compute_reward_variance(trainer):
    """
    Extract per-step reward variance from training logs.
    Returns list of reward margins (chosen - rejected) per logged step.
    """
    margins = []
    for log in trainer.state.log_history:
        if 'rewards/margins' in log:
            margins.append(log['rewards/margins'])
    variance = float(np.var(margins)) if margins else 0.0
    return variance, margins


def compute_implicit_reward_calibration(model, tokenizer, ref_model_logps, dataset_split, num_samples=100, beta=0.1):
    """
    Compute correlation between implicit reward beta*log(pi/pi_ref) and true correctness.
    Approximation: uses log-probs of chosen completions vs a fixed reference.
    """
    samples = list(dataset_split)[:num_samples]
    model.eval()
    implicit_rewards = []
    true_labels = []

    with torch.no_grad():
        for i, item in enumerate(samples):
            prompt = f"Question: {item['question']}\nAnswer:"
            completion = item['answer']
            full_text = prompt + " " + completion

            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=400).to(model.device)
            labels = inputs['input_ids'].clone()

            outputs = model(**inputs, labels=labels)
            log_prob = -outputs.loss.item()  # negative CE = log prob

            # Use stored reference log_prob if available
            ref_lp = ref_model_logps.get(i, log_prob * 0.95)  # fallback approximation
            implicit_reward = beta * (log_prob - ref_lp)
            implicit_rewards.append(implicit_reward)
            true_labels.append(1.0)  # all chosen completions are correct

        # Sample some incorrect completions too
        for i, item in enumerate(samples[:num_samples//2]):
            wrong_item = samples[(i + 1) % len(samples)]
            prompt = f"Question: {item['question']}\nAnswer:"
            completion = wrong_item['answer']  # wrong answer
            full_text = prompt + " " + completion

            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=400).to(model.device)
            labels = inputs['input_ids'].clone()
            outputs = model(**inputs, labels=labels)
            log_prob = -outputs.loss.item()
            ref_lp = ref_model_logps.get(i + 10000, log_prob * 1.05)
            implicit_reward = beta * (log_prob - ref_lp)
            implicit_rewards.append(implicit_reward)
            true_labels.append(0.0)  # incorrect

    correlation = float(np.corrcoef(implicit_rewards, true_labels)[0, 1])
    return correlation


# -------------------------------------------------------------------------
# MAIN EXPERIMENT LOOP
# -------------------------------------------------------------------------

def run_experiment(seed, mode="full", noise_rate=0.0, experiment_tag="e2_7"):
    """Run one full DPO training + evaluation for given seed and mode."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_tag} | Seed: {seed} | Mode: {mode}")
    print(f"{'='*60}")

    set_seed(seed)

    print("Loading dataset...")
    dataset = load_dataset("gsm8k", "main")

    print("Building preference pairs...")
    train_dataset = create_synthetic_dpo_pairs(
        dataset['train'], TRAIN_SAMPLES, mode=mode, noise_rate=noise_rate, seed=seed
    )
    eval_dataset = create_synthetic_dpo_pairs(
        dataset['test'], EVAL_SAMPLES, mode=mode, noise_rate=noise_rate, seed=seed
    )

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    output_dir = os.path.join(OUTPUT_ROOT, f"{experiment_tag}_seed{seed}_mode{mode}")
    os.makedirs(output_dir, exist_ok=True)

    print("Training...")
    trainer = train_dpo(model, tokenizer, train_dataset, eval_dataset, output_dir)

    print("Evaluating accuracy...")
    accuracy = evaluate_accuracy(model, tokenizer, dataset['test'], num_samples=200, seed=seed)

    print("Computing reward variance...")
    reward_variance, reward_margins = compute_reward_variance(trainer)

    # Extract convergence curve (accuracy proxy: reward margin at each eval step)
    convergence_curve = []
    for log in trainer.state.log_history:
        if 'eval_rewards/margins' in log:
            convergence_curve.append({
                'step': log.get('step', 0),
                'eval_margin': log['eval_rewards/margins'],
                'eval_accuracy': log.get('eval_rewards/accuracies', None)
            })

    # Final training loss
    final_loss = None
    for log in reversed(trainer.state.log_history):
        if 'train_loss' in log:
            final_loss = log['train_loss']
            break

    results = {
        'experiment': experiment_tag,
        'seed': seed,
        'mode': mode,
        'noise_rate': noise_rate,
        'final_accuracy': accuracy,
        'reward_variance': reward_variance,
        'reward_margins': reward_margins,
        'convergence_curve': convergence_curve,
        'final_train_loss': final_loss,
        'train_samples': len(train_dataset),
    }

    # Save per-run results
    result_path = os.path.join(OUTPUT_ROOT, f"{experiment_tag}_seed{seed}_mode{mode}_results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {result_path}")
    print(f"  Accuracy: {accuracy:.4f} | Reward Variance: {reward_variance:.4f}")

    # Free memory
    del model, trainer
    torch.cuda.empty_cache()

    return results


def aggregate_results(all_results, tag):
    """Compute mean +/- std across seeds for a set of results."""
    accuracies = [r['final_accuracy'] for r in all_results]
    variances  = [r['reward_variance'] for r in all_results]
    summary = {
        'experiment': tag,
        'n_seeds': len(all_results),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std':  float(np.std(accuracies)),
        'reward_variance_mean': float(np.mean(variances)),
        'reward_variance_std':  float(np.std(variances)),
        'per_seed': all_results
    }
    return summary


# -------------------------------------------------------------------------
# RUN ALL EXPERIMENTS
# -------------------------------------------------------------------------

if __name__ == "__main__":
    all_summaries = {}

    # --- E2.7: Head-to-head, 3 seeds, full labels ---
    print("\n\n*** E2.7: Head-to-Head (3 seeds, full labels) ***")
    e27_results = []
    for seed in SEEDS:
        r = run_experiment(seed=seed, mode="full", noise_rate=0.0, experiment_tag="e2_7")
        e27_results.append(r)
    all_summaries['e2_7'] = aggregate_results(e27_results, 'e2_7')

    # --- E2.9a: Full labels (seed=42 only for speed) ---
    print("\n\n*** E2.9a: Full Labels ***")
    e29_full = run_experiment(seed=42, mode="full", noise_rate=0.0, experiment_tag="e2_9_full")

    # --- E2.9b: Sparse labels (10%) ---
    print("\n\n*** E2.9b: Sparse Labels (10%) ***")
    e29_sparse = run_experiment(seed=42, mode="sparse", noise_rate=0.0, experiment_tag="e2_9_sparse")

    # --- E2.9c: Noisy labels (10% flipped) ---
    print("\n\n*** E2.9c: Noisy Labels (10% flipped) ***")
    e29_noisy = run_experiment(seed=42, mode="noisy", noise_rate=0.1, experiment_tag="e2_9_noisy")

    all_summaries['e2_9'] = {
        'full':   e29_full,
        'sparse': e29_sparse,
        'noisy':  e29_noisy,
    }

    # --- Save master results file ---
    master_path = os.path.join(OUTPUT_ROOT, "dpo_all_results.json")
    with open(master_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    # --- Print final summary table ---
    print("\n\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    e27 = all_summaries['e2_7']
    print(f"\nE2.7 DPO on GSM8K (n={e27['n_seeds']} seeds):")
    print(f"  Accuracy:        {e27['accuracy_mean']:.4f} ± {e27['accuracy_std']:.4f}")
    print(f"  Reward Variance: {e27['reward_variance_mean']:.4f} ± {e27['reward_variance_std']:.4f}")

    print(f"\nE2.9 Label Regime Comparison:")
    for regime, res in all_summaries['e2_9'].items():
        print(f"  {regime:8s}: accuracy={res['final_accuracy']:.4f}, reward_var={res['reward_variance']:.4f}")

    print(f"\nAll results saved to: {master_path}")
    print("="*60)
