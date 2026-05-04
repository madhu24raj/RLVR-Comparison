"""Tests locking in the cluster-readiness fixes."""
import sys
import os
import inspect
import argparse

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestCLIFlags:
    """Both run scripts must accept the new cluster CLI flags."""

    def test_run_e2_7_has_model_name_flag(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "ppo_specs" / "run_e2_7.py").read_text()
        assert "--model-name" in src

    def test_run_e2_7_has_mitigation_flags(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "ppo_specs" / "run_e2_7.py").read_text()
        for flag in ("--gradient-checkpointing", "--optimizer-8bit",
                     "--optimizer-fused", "--reference-quant",
                     "--length-bucketed-generation"):
            assert flag in src, f"{flag} missing from run_e2_7.py"

    def test_run_e2_8_has_model_name_and_mitigations(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "ppo_specs" / "run_e2_8.py").read_text()
        for flag in ("--model-name", "--gradient-checkpointing",
                     "--optimizer-8bit", "--optimizer-fused",
                     "--reference-quant", "--length-bucketed-generation",
                     "--checkpoint-every", "--resume-from"):
            assert flag in src, f"{flag} missing from run_e2_8.py"


class TestConfigHashExpanded:
    """_config_hash must include reward/KL/PPO-epoch fields to prevent
    silent resume mismatches."""

    def test_hash_includes_reward_mode(self):
        from ppo_specs.config import PPOConfig
        from ppo_specs.checkpoint import _config_hash
        cfg_a = PPOConfig(reward_mode="deterministic")
        cfg_b = PPOConfig(reward_mode="self_judge")
        assert _config_hash(cfg_a) != _config_hash(cfg_b)

    def test_hash_includes_reference_kl_coeff(self):
        from ppo_specs.config import PPOConfig
        from ppo_specs.checkpoint import _config_hash
        cfg_a = PPOConfig(reference_kl_coeff=0.0)
        cfg_b = PPOConfig(reference_kl_coeff=0.01)
        assert _config_hash(cfg_a) != _config_hash(cfg_b)

    def test_hash_includes_n_ppo_epochs(self):
        from ppo_specs.config import PPOConfig
        from ppo_specs.checkpoint import _config_hash
        cfg_a = PPOConfig(n_ppo_epochs=1)
        cfg_b = PPOConfig(n_ppo_epochs=4)
        assert _config_hash(cfg_a) != _config_hash(cfg_b)

    def test_hash_stable_across_irrelevant_changes(self):
        from ppo_specs.config import PPOConfig
        from ppo_specs.checkpoint import _config_hash
        cfg_a = PPOConfig(n_steps=100)
        cfg_b = PPOConfig(n_steps=200)
        assert _config_hash(cfg_a) == _config_hash(cfg_b)


class TestEvaluateSelfJudge:
    """evaluate() must call set_questions() so the self-judge wrapper
    has matching question context."""

    def test_evaluate_source_calls_set_questions(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.evaluate)
        assert "set_questions" in src, (
            "evaluate() must call set_questions() on the reward_fn for the "
            "self-judge wrapper, otherwise reward_mode='self_judge' crashes "
            "during eval."
        )


class TestEvaluateShardPadding:
    """evaluate() under DDP must pad to a multiple of world_size to avoid
    silently dropping samples when n_eval is not divisible."""

    def test_evaluate_handles_non_divisible_n_eval(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.evaluate)
        # Heuristic: source mentions padding to handle divisibility,
        # OR explicitly asserts divisibility with a clear error.
        has_padding = "pad_count" in src or "pad_eval" in src
        has_assertion = "assert" in src and "ws" in src
        assert has_padding or has_assertion, (
            "evaluate() must either pad eval set to multiple of world_size "
            "or assert divisibility — silently dropping samples is unsafe."
        )


class TestRunE2_8SetsQuestions:
    """run_e2_8.py must call set_questions() before train_step so self-judge
    works in E2.8 sweeps."""

    def test_run_e2_8_calls_set_questions(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "ppo_specs" / "run_e2_8.py").read_text()
        assert "set_questions" in src, (
            "run_e2_8.py must call reward_fn.set_questions() before "
            "train_step so reward_mode='self_judge' works in E2.8 sweeps."
        )


class TestGracefulExitHandlerSIGUSR1:
    """GracefulExitHandler must listen for SIGUSR1 (SLURM preempt signal)."""

    def test_handler_registers_sigusr1(self):
        import inspect
        from ppo_specs.checkpoint import GracefulExitHandler
        src = inspect.getsource(GracefulExitHandler)
        assert "SIGUSR1" in src, (
            "GracefulExitHandler must register SIGUSR1 handler for SLURM "
            "preemption (with hasattr guard for Windows)."
        )


class TestMainProcessFirst:
    """load_ppo_trainer must use accelerator.main_process_first() to avoid
    concurrent HF cache thrash on multi-rank first run."""

    def test_load_uses_main_process_first(self):
        from ppo_specs import ppo_trainer
        src = inspect.getsource(ppo_trainer.load_ppo_trainer)
        assert "main_process_first" in src, (
            "load_ppo_trainer must wrap from_pretrained() in "
            "accelerator.main_process_first() to avoid concurrent "
            "HF download/cache thrash."
        )


class TestSlurmScriptFixes:
    def test_sigusr1_forward_uncommented(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "scripts" / "slurm_e2_7.sh").read_text()
        # The active line must NOT be commented out
        assert "kill -SIGUSR1" in src
        # Verify it's not just in a comment by checking for an active line
        active_lines = [
            line for line in src.splitlines()
            if "kill -SIGUSR1" in line and not line.strip().startswith("#")
        ]
        assert active_lines, "kill -SIGUSR1 must be uncommented in handle_preempt"

    def test_mem_bumped_to_128g(self):
        from pathlib import Path
        for fname in ("slurm_e2_7.sh", "slurm_e2_8.sh"):
            src = (Path(_PROJECT_ROOT) / "scripts" / fname).read_text()
            assert "--mem=128G" in src, f"{fname} should have --mem=128G for 8B"

    def test_partition_expansion_removed(self):
        from pathlib import Path
        for fname in ("slurm_e2_7.sh", "slurm_e2_8.sh"):
            src = (Path(_PROJECT_ROOT) / "scripts" / fname).read_text()
            assert "${PARTITION:-gpu}" not in src, (
                f"{fname}: #SBATCH --partition=${{PARTITION:-gpu}} uses bash "
                "expansion that Slurm doesn't process. Remove the directive "
                "and use 'sbatch -p ...' instead."
            )

    def test_model_name_passed_to_python(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "scripts" / "slurm_e2_7.sh").read_text()
        assert "--model-name" in src, (
            "slurm_e2_7.sh must pass --model-name $MODEL_NAME to Python; "
            "the env var alone doesn't reach the script."
        )

    def test_mitigation_flags_passed_for_8b(self):
        from pathlib import Path
        src = (Path(_PROJECT_ROOT) / "scripts" / "slurm_e2_7.sh").read_text()
        assert "--gradient-checkpointing" in src
        assert "--optimizer-8bit" in src


class TestRequirements:
    def test_bitsandbytes_in_requirements(self):
        from pathlib import Path
        req = (Path(_PROJECT_ROOT) / "requirements.txt").read_text()
        assert "bitsandbytes" in req.lower(), (
            "bitsandbytes must be in requirements.txt for the 8B path "
            "(optimizer_8bit and reference_quant)."
        )
