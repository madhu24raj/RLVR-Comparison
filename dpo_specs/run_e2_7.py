import sys
import os
import argparse
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import load_gsm8k, format_prompt
from src.rewards import gsm8k_reward
from eval.metrics import ExperimentLogger
from ppo_specs.utils import cycle_batch
from ppo_specs.config import local_test_config, e2_7_config

from dpo_specs.dpo_trainer import IterativeDPOTrainer

def run_dpo_e2_7(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_seed(config.seed)
    
    print(f"[E2.7 DPO] Loading Data...")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=200)

    train_prompts = [format_prompt(ex["question"]) for ex in train_ds]
    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_prompts  = [format_prompt(ex["question"]) for ex in test_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    print(f"[E2.7 DPO] Loading Models...")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # DPO requires both a policy model and a frozen reference model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch_dtype).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch_dtype).to(device)
    ref_model.eval()

    trainer = IterativeDPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=gsm8k_reward,
        device=device
    )

    logger = ExperimentLogger(f"{config.experiment_name}_DPO", config.output_dir)
    reward_window = []

    print("[E2.7 DPO] Starting Training Loop...")
    for step in range(config.n_steps):
        batch_p  = cycle_batch(train_prompts, step, config.batch_size)
        batch_gt = cycle_batch(train_gts,     step, config.batch_size)

        metrics = trainer.train_step(batch_p, batch_gt)
        reward_window.append(metrics["mean_reward"])

        if step % config.log_every == 0:
            print(
                f"  step {step:3d} | reward={metrics['mean_reward']:.3f} "
                f"| acc={metrics['accuracy']:.3f} "
                f"| dpo_loss={metrics['dpo_loss']:.4f} "
                f"| pairs={metrics['valid_pairs']}"
            )

        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=20)
            window = reward_window[-config.eval_every:] if len(reward_window) >= config.eval_every else reward_window
            stability = float(np.var(window))

            logger.log_step(
                step,
                total_rollouts=metrics["total_rollouts"],
                train_accuracy=metrics["accuracy"],
                test_accuracy=test_acc,
                reward_variance=stability,
                dpo_loss=metrics["dpo_loss"],
                valid_pairs=metrics["valid_pairs"]
            )
            logger.save()
            print(f"    -> test_acc={test_acc:.3f} | stability(var)={stability:.4f}")

    logger.save()
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=50)
    print(f"\n[E2.7 DPO] Final test accuracy: {final_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2.7: DPO head-to-head on GSM8K")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    run_dpo_e2_7(cfg)