"""Regression test for the P12 epoch-0 skip × capacity='none' interaction.

Bug history: when n_ppo_epochs >= 2 and capacity == "none" and
reference_kl_coeff == 0, the epoch-0 ppo_update produces a total_loss
with no gradient (policy is detached, critic is non-trainable, kl_ref is
disabled). Calling .backward() on it raises:
    RuntimeError: element 0 of tensors does not require grad and does not
    have a grad_fn.

Fix: ppo_update guards backward()/step() behind total_loss.requires_grad.
This test asserts the guard is in place and that an end-to-end train_step
on the REINFORCE path completes cleanly.
"""
import sys
import os
import inspect

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestNoGradBackwardGuard:
    def test_ppo_update_source_guards_backward_on_requires_grad(self):
        """Static check: ppo_update must guard backward() / step() behind
        total_loss.requires_grad to avoid the REINFORCE epoch-0 crash."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.ppo_update)
        assert "total_loss.requires_grad" in src, (
            "ppo_update must check total_loss.requires_grad before calling "
            "backward(). Without the guard, P12 epoch-0 skip with "
            'capacity="none" crashes with "element 0 of tensors does not '
            'require grad" because policy_loss/critic_loss/kl all detach.'
        )

    @pytest.mark.slow
    def test_train_step_completes_with_capacity_none_and_K2(self):
        """End-to-end: REINFORCE (capacity='none') with n_ppo_epochs>=2
        must run train_step without crashing."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ppo_specs.config import PPOConfig
        from ppo_specs.ppo_trainer import PPOTrainer
        from ppo_specs.critic import build_critic
        from src.rewards import gsm8k_reward

        name = "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name).to(torch.device("cpu"))

        cfg = PPOConfig(
            model_name=name,
            critic_capacity="none",
            n_ppo_epochs=2,         # P12 trigger
            reference_kl_coeff=0.0, # no KL anchor → no extra grad
            kl_coeff=0.0,
            batch_size=2, max_new_tokens=4, max_prompt_length=32,
            do_sample=False,
            n_steps=1, eval_every=1, log_every=1,
            checkpoint_every=0,
            experiment_name="test_reinforce_K2",
        )
        critic = build_critic("none", model.config.hidden_size).to(torch.device("cpu"))
        trainer = PPOTrainer(
            config=cfg, model=model, tokenizer=tok, critic=critic,
            reward_fn=gsm8k_reward, device=torch.device("cpu"),
        )

        prompts = ["What is 2+2?", "Solve 1+1="]
        gts = ["4", "2"]
        # Should complete without RuntimeError
        metrics = trainer.train_step(prompts, gts)
        # Sanity: standard metrics present and finite
        import math
        for key in ("policy_loss", "critic_loss", "kl_divergence",
                    "mean_reward", "accuracy"):
            assert key in metrics
            assert math.isfinite(metrics[key]), f"Bad metric {key}={metrics[key]}"
