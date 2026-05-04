"""Tests for ppo_trainer.py optimizations: AdamW variants, P12, P15, P18.

Most tests use mocks/synthetic inputs to avoid loading a real LM.
"""
import sys
import os
import inspect
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig
from ppo_specs.ppo_trainer import _build_adamw


class TestAdamWBuilder:
    """Test the _build_adamw helper that selects optimizer variants."""

    def test_default_returns_torch_adamw(self):
        params = [nn.Parameter(torch.randn(5))]
        opt = _build_adamw(params, lr=1e-4, use_8bit=False, use_fused=False)
        assert isinstance(opt, torch.optim.AdamW)

    def test_fused_falls_back_on_cpu(self):
        """fused=True is CUDA-only; CPU path must fall back gracefully."""
        params = [nn.Parameter(torch.randn(5))]
        # Should not raise even though fused=True is requested on CPU
        opt = _build_adamw(params, lr=1e-4, use_8bit=False, use_fused=True)
        assert isinstance(opt, torch.optim.AdamW)

    def test_8bit_falls_back_when_bnb_missing(self):
        """If bitsandbytes is not importable, fall back to torch AdamW."""
        params = [nn.Parameter(torch.randn(5))]
        # Test will work whether bnb is installed or not. Just verify
        # the function returns SOMETHING that quacks like an optimizer.
        opt = _build_adamw(params, lr=1e-4, use_8bit=True, use_fused=False)
        assert hasattr(opt, "step")
        assert hasattr(opt, "zero_grad")


class TestExtractLastHiddenHook:
    """Verify _extract_last_hidden uses a forward hook on the FINAL NORM
    so it matches output_hidden_states[-1] bitwise (P15)."""

    def test_uses_forward_hook(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer._extract_last_hidden)
        assert "register_forward_hook" in src, (
            "_extract_last_hidden must register a forward hook (P15) "
            "as the primary code path."
        )

    def test_hooks_final_norm_not_decoder_layer(self):
        """P15 correctness: hooking model.layers[-1] returns PRE-norm
        activations, which differ from output_hidden_states[-1] by
        the final RMSNorm. The hook must target the final norm so the
        captured tensor matches output_hidden_states[-1] bitwise."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer._extract_last_hidden)
        assert "_get_final_norm_layer" in src, (
            "_extract_last_hidden must call _get_final_norm_layer (not "
            "_get_last_decoder_layer) so the captured hidden state matches "
            "output_hidden_states[-1] bitwise. Hooking the last decoder "
            "block returns PRE-norm activations (max-abs-diff ~163 on "
            "Qwen2.5-0.5B)."
        )

    def test_get_final_norm_layer_helper_exists(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        assert hasattr(PPOTrainer, "_get_final_norm_layer")

    def test_get_final_norm_locates_qwen_norm(self):
        """For Llama/Qwen/Mistral, _get_final_norm_layer should find
        m.model.norm and return it."""
        from ppo_specs.ppo_trainer import PPOTrainer

        # Build a tiny mock that mimics Llama/Qwen layout
        mock_norm = nn.LayerNorm(8)
        mock_inner = nn.Module()
        mock_inner.norm = mock_norm
        mock_outer = nn.Module()
        mock_outer.model = mock_inner

        # Bind PPOTrainer._get_final_norm_layer to a synthetic self
        class _Stub:
            pass
        stub = _Stub()
        stub.model = mock_outer
        result = PPOTrainer._get_final_norm_layer(stub)
        assert result is mock_norm, (
            f"Expected to find m.model.norm, got {result}"
        )

    def test_fallback_when_norm_not_locatable(self):
        """For unknown model architectures, _get_final_norm_layer
        returns None and the caller falls back to output_hidden_states=True."""
        from ppo_specs.ppo_trainer import PPOTrainer

        class _Stub:
            pass
        stub = _Stub()
        stub.model = nn.Linear(4, 4)  # not a HF causal LM
        result = PPOTrainer._get_final_norm_layer(stub)
        assert result is None

    @pytest.mark.slow
    def test_hook_matches_output_hidden_states_bitwise(self):
        """Real-model correctness check: hooking the final norm must
        produce the same hidden state as output_hidden_states[-1]."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        m = AutoModelForCausalLM.from_pretrained(model_name)
        m.eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        prompts = ["What is 2+2?", "Solve: 5 * 3 ="]
        enc = tok(prompts, return_tensors="pt", padding=True, max_length=64, truncation=True)

        # Baseline: output_hidden_states[-1]
        with torch.no_grad():
            out = m(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=True, use_cache=False,
            )
        baseline = out.hidden_states[-1]

        # Hook on m.model.norm (final RMSNorm)
        captured = {}
        def _hook(mod, inp, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output

        handle = m.model.norm.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = m(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    use_cache=False,
                )
        finally:
            handle.remove()
        hooked = captured["h"]

        torch.testing.assert_close(
            hooked, baseline, atol=0.0, rtol=0.0,
            msg="Hook on final norm must produce bitwise-identical "
                "output to output_hidden_states[-1]",
        )


class TestVectorizedItemCalls:
    """Verify P18: no per-i .item() loops in generate_rollouts."""

    def test_generate_rollouts_no_per_loop_item_for_logprobs(self):
        """The current impl had `r.old_log_prob = old_log_probs[i].item()`
        inside a Python for-loop. After P18 it should be a vectorized
        cpu().tolist() before the loop."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.generate_rollouts)
        # Heuristic: there should NOT be `[i].item()` patterns for
        # old_log_probs or critic_values inside the loop body.
        assert "old_log_probs[i].item()" not in src, (
            "P18 violation: per-sample .item() in old_log_probs loop"
        )
        assert "critic_values[i].item()" not in src, (
            "P18 violation: per-sample .item() in critic_values loop"
        )


class TestEpochZeroSkip:
    """Verify P12: ppo_update accepts is_first_epoch flag."""

    def test_ppo_update_signature_includes_is_first_epoch(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        sig = inspect.signature(PPOTrainer.ppo_update)
        assert "is_first_epoch" in sig.parameters, (
            "ppo_update must accept is_first_epoch flag (P12) so the "
            "first PPO epoch can skip the redundant policy forward."
        )

    def test_train_step_passes_is_first_epoch(self):
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step)
        # Must call ppo_update with is_first_epoch kwarg
        assert "is_first_epoch" in src, (
            "train_step must pass is_first_epoch to ppo_update."
        )


class TestQuantizationConfigPath:
    """Verify the reference_quant config knob is wired through load_ppo_trainer."""

    def test_load_ppo_trainer_reads_reference_quant(self):
        from ppo_specs import ppo_trainer
        src = inspect.getsource(ppo_trainer.load_ppo_trainer)
        assert "reference_quant" in src, (
            "load_ppo_trainer must read config.reference_quant to "
            "support int8/nf4 quantization of the frozen reference model."
        )

    def test_load_ppo_trainer_handles_bnb_import_failure(self):
        """When bitsandbytes is missing, the code path must not crash."""
        from ppo_specs import ppo_trainer
        src = inspect.getsource(ppo_trainer.load_ppo_trainer)
        # Must have a try/except around the bnb import
        assert "ImportError" in src or "BitsAndBytesConfig" in src, (
            "load_ppo_trainer must handle the case where bitsandbytes "
            "is not installed."
        )
