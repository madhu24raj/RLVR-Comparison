import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import dataclasses
import random

import torch
import numpy as np
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as transformers_set_seed
from trl import DPOTrainer, DPOConfig as TRLDPOConfig
from datasets import Dataset

from src.tasks import get_task
from src.dpo_pairs import construct_pairs_from_batch, pairs_to_dataset
from eval.metrics import accuracy as compute_accuracy
from eval.metrics import ExperimentLogger
from dpo_specs.config import DPOConfig, local_test_config, e2_7_config
from grpo_specs.checkpoint import (
    save_grpo_checkpoint, load_grpo_checkpoint,
    find_latest_checkpoint, restore_rng_states, GracefulExitHandler,
)


# TRL's DPOConfig field set drifts between versions (e.g. recent releases dropped
# / renamed `max_prompt_length`, `max_length`). Build the config from a superset
# of desired kwargs but only pass those the *installed* version actually accepts,
# so the driver runs unchanged across TRL versions on Colab / the cluster.
_DPO_CONFIG_FIELDS = {f.name for f in dataclasses.fields(TRLDPOConfig)}


def _make_trl_dpo_config(**desired):
    accepted = {k: v for k, v in desired.items() if k in _DPO_CONFIG_FIELDS}
    dropped = [k for k in desired if k not in _DPO_CONFIG_FIELDS]
    if dropped:
        print(f"[DPO] Installed TRL DPOConfig ignores {dropped} "
              f"(version drift); proceeding without them.")
    return TRLDPOConfig(**accepted)

class IterativeDPOTrainer:
    def __init__(self, config, model, ref_model, tokenizer, reward_fn, device):
        self.config = config
        self.model = model
        self.ref_model = ref_model  # DPO requires a frozen reference model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = device
        
        self.total_rollouts = 0
        self.step = 0

    def train_step(self, prompts: List[str], ground_truths: List[str]) -> Dict[str, float]:
        self.model.eval()
        
        # CRITICAL FIX: DPO needs multiple completions per prompt to form chosen/rejected pairs!
        n_rollouts = getattr(self.config, "n_rollouts_per_prompt", 4)
        expanded_prompts = [p for p in prompts for _ in range(n_rollouts)]
        expanded_gts = [gt for gt in ground_truths for _ in range(n_rollouts)]
        
        completions = []
        rewards = []
        
        # 1. Generate rollouts (Micro-batched to prevent GPU OOM)
        chunk_size = 8 
        for i in range(0, len(expanded_prompts), chunk_size):
            chunk_p = expanded_prompts[i:i+chunk_size]
            chunk_gt = expanded_gts[i:i+chunk_size]
            
            enc = self.tokenizer(
                chunk_p, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            
            for j in range(len(chunk_p)):
                pl = prompt_lens[j]
                pad_len = (enc["input_ids"][j] == self.tokenizer.pad_token_id).sum().item()
                real_start = pad_len
                
                completion = self.tokenizer.decode(out[j][real_start + pl:], skip_special_tokens=True)
                completions.append(completion)
                rewards.append(self.reward_fn(completion, chunk_gt[j]))

        self.total_rollouts += len(expanded_prompts)
        
        # 2. Construct DPO Preference Pairs
        pairs = construct_pairs_from_batch(
            expanded_prompts, completions, rewards, strategy="all", seed=self.config.seed + self.step
        )
        
        metrics = {
            "mean_reward": float(np.mean(rewards)),
            "reward_variance": float(np.var(rewards)) if len(rewards) > 1 else 0.0,
            "accuracy": compute_accuracy(rewards),
            "total_rollouts": self.total_rollouts,
            "valid_pairs": len(pairs),
            "dpo_loss": 0.0,
            "kl_ref_divergence": 0.0, # Added for tracking
            "kl_divergence": 0.0      # Mirrored so the PPO logging script doesn't crash
        }
        
        # 3. DPO Update (if valid pairs exist)
        if pairs:
            self.model.train()
            pair_dict = pairs_to_dataset(pairs)
            dataset = Dataset.from_dict(pair_dict)
            
            dpo_config = _make_trl_dpo_config(
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=max(1, len(pairs) // 2),
                max_length=1024,
                max_prompt_length=512,
                beta=getattr(self.config, "beta", 0.1), # DPO KL penalty parameter
                report_to="none",
                remove_unused_columns=False,
                logging_steps=1, # CRITICAL: Forces TRL to save metrics to its log_history
            )
            
            # TRL renamed `tokenizer=` -> `processing_class=` around 0.12. Try the
            # newer name first (Colab/recent), fall back to the old one.
            _dpo_kwargs = dict(
                model=self.model,
                ref_model=self.ref_model,
                args=dpo_config,
                train_dataset=dataset,
            )
            try:
                trainer = DPOTrainer(**_dpo_kwargs, processing_class=self.tokenizer)
            except TypeError:
                trainer = DPOTrainer(**_dpo_kwargs, tokenizer=self.tokenizer)
            
            train_result = trainer.train()
            metrics["dpo_loss"] = train_result.training_loss
            
            # 4. Extract KL Divergence from TRL's internal logs
            _beta = getattr(self.config, "beta", 0.1)
            if trainer.state.log_history:
                for log in reversed(trainer.state.log_history):
                    if "rewards/chosen" in log:
                        # TRL logs 'rewards/chosen' which mathematically is: beta * (log_pi - log_ref)
                        # Dividing by beta isolates the pure KL divergence!
                        kl_ref = float(log["rewards/chosen"] / _beta)
                        metrics["kl_ref_divergence"] = kl_ref
                        metrics["kl_divergence"] = kl_ref
                        break
            
        self.step += 1
        return metrics

    @torch.no_grad()
    def evaluate(self, prompts: List[str], ground_truths: List[str], n_eval: int = 50) -> float:
        self.model.eval()
        eval_prompts = prompts[:n_eval]
        eval_gts = ground_truths[:n_eval]
        
        enc = self.tokenizer(eval_prompts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False, # Greedy
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        rewards = []
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i in range(len(eval_prompts)):
            pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
            completion = self.tokenizer.decode(out[i][pad_len + prompt_lens[i]:], skip_special_tokens=True)
            rewards.append(self.reward_fn(completion, eval_gts[i]))

        return compute_accuracy(rewards)


# ── Trainer factory ───────────────────────────────────────────────────────────

def load_dpo_trainer(config, device, model_path_override=None):
    """Load policy + frozen reference model + tokenizer, build the trainer.

    DPO always needs a frozen reference. On resume the POLICY loads from
    model_path_override (a checkpoint dir); the reference always loads from
    config.model_name (the initial policy, the correct DPO anchor).
    """
    if config.torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    policy_src = model_path_override or config.model_name
    print(f"[DPO] Loading model: {policy_src} (device={device}, dtype={torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(policy_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        policy_src, torch_dtype=torch_dtype,
    ).to(device)
    if getattr(config, "_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[DPO] Gradient checkpointing enabled")

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch_dtype,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    task = get_task(config.task)
    print(f"[DPO] Task: {task.name}")

    trainer = IterativeDPOTrainer(
        config, model, ref_model, tokenizer, reward_fn=task.reward, device=device,
    )
    return trainer, task


def cycle_batch(items, step, batch_size):
    """Deterministic cycling through items by step (matches PPO/GRPO)."""
    n = len(items)
    start = (step * batch_size) % n
    end = start + batch_size
    if end <= n:
        return items[start:end]
    return items[start:] + items[:end - n]


def run_dpo(config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    transformers_set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"[DPO] Device: {device}")

    # -- Data (task abstraction: gsm8k both splits; humaneval = MBPP -> HumanEval) --
    data_task = get_task(config.task)
    print(f"[DPO] Task: {data_task.name} | Loading data ...")
    train_ds = data_task.load("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = data_task.load(data_task.eval_split, n_samples=config.n_test_samples, seed=config.seed)
    train_gts = [ex["ground_truth"] for ex in train_ds]
    test_gts  = [ex["ground_truth"] for ex in test_ds]

    # -- Resume? --
    ckpt_dir = f"{config.checkpoint_dir}/{config.experiment_name}"
    resume_state = None
    model_path_override = None
    resume_path = config.resume_from
    if resume_path:
        if resume_path == "auto":
            resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[DPO] Resuming from {resume_path}")
            resume_state = load_grpo_checkpoint(resume_path, device)
            model_path_override = resume_state["model_path"]
        else:
            print(f"[DPO] No checkpoint found in {ckpt_dir}; starting fresh.")

    trainer, task = load_dpo_trainer(config, device, model_path_override=model_path_override)

    train_prompts = [task.format_prompt(ex, trainer.tokenizer) for ex in train_ds]
    test_prompts  = [task.format_prompt(ex, trainer.tokenizer) for ex in test_ds]

    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_window: List[float] = []
    start_step = 0
    if resume_state is not None:
        trainer.total_rollouts = resume_state["total_rollouts"]
        trainer.step = resume_state["step"] + 1
        logger.log = resume_state["logger_log"]
        restore_rng_states(resume_state["rng_states"])
        start_step = resume_state["step"] + 1
        print(f"[DPO] Resumed at step {start_step}")

    exit_handler = GracefulExitHandler()

    def _checkpoint(step, keep):
        if config.checkpoint_every <= 0:
            return
        save_grpo_checkpoint(trainer, step, config, logger, ckpt_dir, keep)

    for step in range(start_step, config.n_steps):
        batch_p  = cycle_batch(train_prompts, step, config.batch_size)
        batch_gt = cycle_batch(train_gts,     step, config.batch_size)

        metrics = trainer.train_step(batch_p, batch_gt)
        reward_window.append(metrics["mean_reward"])

        if step % config.log_every == 0:
            print(
                f"  step {step:3d} | reward={metrics['mean_reward']:.3f} "
                f"| acc={metrics['accuracy']:.3f} | pairs={metrics['valid_pairs']} "
                f"| dpo_loss={metrics['dpo_loss']:.4f} | kl={metrics['kl_divergence']:.4f}"
            )

        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.eval_size)
            window = reward_window[-config.eval_every:] if len(reward_window) >= config.eval_every \
                     else reward_window
            stability = float(np.var(window))
            logger.log_step(step, **{
                "total_rollouts":   metrics["total_rollouts"],
                "train_accuracy":   metrics["accuracy"],
                "test_accuracy":    test_acc,
                "reward_variance":  stability,
                "dpo_loss":         metrics["dpo_loss"],
                "valid_pairs":      metrics["valid_pairs"],
                "kl_divergence":    metrics["kl_divergence"],
                "kl_ref_divergence": metrics["kl_ref_divergence"],
            })
            logger.save()
            print(f"    -> test_acc={test_acc:.3f} | stability(var)={stability:.4f}")

        if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            _checkpoint(step, keep=config.keep_checkpoints)

        if exit_handler.should_exit:
            print(f"[DPO] Graceful exit requested at step {step}; saving checkpoint.")
            _checkpoint(step, keep=0)
            logger.save()
            return

    logger.save()
    _checkpoint(config.n_steps - 1, keep=0)

    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.final_eval_size)
    print(f"\n[DPO] Final test accuracy: {final_acc:.3f}")
    print(f"[DPO] Log saved to {config.output_dir}/{config.experiment_name}.json")
    print(f"[DPO] Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative DPO head-to-head (E2.7)")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model (e.g. Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["gsm8k", "humaneval"],
                        help="Task: gsm8k (default) or humaneval (MBPP train -> HumanEval eval)")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--keep-checkpoints", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint dir to resume from, or 'auto' for the latest")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write the metrics JSON (point at Drive on Colab)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    if args.task is not None:
        cfg.task = args.task
    if args.model_name:
        cfg.model_name = args.model_name
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
        cfg.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.keep_checkpoints is not None:
        cfg.keep_checkpoints = args.keep_checkpoints
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg._gradient_checkpointing = args.gradient_checkpointing

    run_dpo(cfg)