import torch
import numpy as np
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from src.rewards import gsm8k_reward
from src.dpo_pairs import construct_pairs_from_batch, pairs_to_dataset
from eval.metrics import accuracy as compute_accuracy

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
        n_rollouts = 4  # Generate 4 attempts per prompt to create contrastive pairs
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
            
            dpo_config = DPOConfig(
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=max(1, len(pairs) // 2), 
                max_length=1024,
                max_prompt_length=512,
                beta=0.1, # DPO KL penalty parameter
                report_to="none",
                remove_unused_columns=False,
                logging_steps=1, # CRITICAL: Forces TRL to save metrics to its log_history
            )
            
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=dpo_config,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )
            
            train_result = trainer.train()
            metrics["dpo_loss"] = train_result.training_loss
            
            # 4. Extract KL Divergence from TRL's internal logs
            if trainer.state.log_history:
                for log in reversed(trainer.state.log_history):
                    if "rewards/chosen" in log:
                        # TRL logs 'rewards/chosen' which mathematically is: beta * (log_pi - log_ref)
                        # Dividing by beta isolates the pure KL divergence!
                        kl_ref = float(log["rewards/chosen"] / dpo_config.beta)
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