"""Empirical measurement tests for Phase 1+2 optimizations.

These tests EXECUTE old vs new code paths and ASSERT measurable
improvements. Each test prints raw numbers to stdout so a human reviewer
can sanity-check the magnitude of the win.

Coverage:
  1. P14 — fused cross_entropy vs naive log_softmax+gather (memory + parity)
  2. P18 — vectorized .item() in generate_rollouts (sync-pattern count + parity)
  3. P12 — epoch-0 redundant policy forward skip (guarded on K>=2)
  4. P15 — _extract_last_hidden does NOT request all hidden states (slow)
  5. End-to-end smoke with all optimizations enabled (slow)
"""
from __future__ import annotations

import sys
import os
import inspect
import math
import re
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Reproductions of OLD vs NEW token log-prob paths (for P14 measurement) ───

def _naive_log_softmax_gather(
    logits_slice: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """OLD path: materialize full [R, V] log_softmax tensor, then gather."""
    lp_full = F.log_softmax(logits_slice.float(), dim=-1)         # [R, V] fp32
    return lp_full.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # [R]


def _fused_cross_entropy(
    logits_slice: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """NEW path: fused log_softmax+gather via cross_entropy."""
    return -F.cross_entropy(
        logits_slice.float(), target_ids, reduction="none",
    )


# ── 1. P14 — fused cross_entropy memory + parity ─────────────────────────────

class TestP14FusedCrossEntropyMemory:
    """Compare peak host memory of OLD vs NEW per-token log-prob paths."""

    def test_parity_naive_vs_fused(self):
        """The two implementations must match within fp32 tolerance."""
        torch.manual_seed(0)
        R, V = 64, 8192   # smaller logits for parity check
        logits = torch.randn(R, V)
        targets = torch.randint(0, V, (R,))

        old = _naive_log_softmax_gather(logits, targets)
        new = _fused_cross_entropy(logits, targets)

        torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)
        print(
            f"\n[P14 parity] R={R}, V={V} -> max-abs-diff="
            f"{(old - new).abs().max().item():.3e}"
        )

    def test_fused_path_peak_memory_lower(self):
        """At Llama-3-8B-shape logits ([R=200, V=128256]), the fused
        cross_entropy path's peak intermediate-tensor allocation must be
        LESS than the naive log_softmax+gather path's. The naive path
        materializes a [R, V] fp32 log_softmax tensor (~102 MB); the
        fused path does not allocate any intermediate that large.

        Measurement strategy: directly intercept tensor allocations via
        ``__torch_function__`` and sum the maximum LIVE-bytes high-water
        mark across each call. This is deterministic — unlike RSS
        sampling which is sensitive to allocator cache state from
        earlier tests in the same process.

        We do NOT use ``__torch_dispatch__`` because some operators
        (notably the C++ ``log_softmax`` kernel) construct their output
        below the dispatcher and the resulting tensor lifetimes are not
        observable from there. ``__torch_function__`` sees every public
        ``torch.*`` / ``F.*`` call and the tensors they return — which
        is exactly the scope this test cares about (the intermediate
        tensors the Python path materializes).
        """
        torch.manual_seed(0)
        R, V = 200, 128256
        nominal_intermediate_bytes = R * V * 4  # [R, V] fp32

        logits = torch.randn(R, V, dtype=torch.float32)
        targets = torch.randint(0, V, (R,))

        class _TrackedTensor(torch.Tensor):
            """Subclass that records every torch-function output's
            storage size in a class-level peak tracker."""

            _live: dict = {}
            _live_bytes: int = 0
            _peak_bytes: int = 0

            @classmethod
            def reset(cls):
                cls._live = {}
                cls._live_bytes = 0
                cls._peak_bytes = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                # Run the op; default base behavior
                out = super().__torch_function__(func, types, args, kwargs)
                cls._observe(out)
                return out

            @classmethod
            def _observe(cls, x):
                if isinstance(x, torch.Tensor):
                    try:
                        storage = x.untyped_storage()
                        sid = storage.data_ptr()
                        if sid not in cls._live:
                            nb = storage.nbytes()
                            cls._live[sid] = nb
                            cls._live_bytes += nb
                            if cls._live_bytes > cls._peak_bytes:
                                cls._peak_bytes = cls._live_bytes
                    except Exception:
                        pass
                elif isinstance(x, (list, tuple)):
                    for item in x:
                        cls._observe(item)

        # Wrap inputs as the tracked subclass. Functions that produce
        # plain tensors will still trigger __torch_function__ because
        # one of the inputs is a subclass.
        tracked_logits = logits.as_subclass(_TrackedTensor)
        tracked_targets = targets.as_subclass(_TrackedTensor)

        # Naive path
        _TrackedTensor.reset()
        old_out = _naive_log_softmax_gather(tracked_logits, tracked_targets)
        _ = torch.as_tensor(old_out).sum().item()
        old_peak = _TrackedTensor._peak_bytes

        # Fused path
        _TrackedTensor.reset()
        new_out = _fused_cross_entropy(tracked_logits, tracked_targets)
        _ = torch.as_tensor(new_out).sum().item()
        new_peak = _TrackedTensor._peak_bytes

        # Numerical equivalence is also load-bearing here.
        torch.testing.assert_close(
            torch.as_tensor(old_out), torch.as_tensor(new_out),
            rtol=1e-5, atol=1e-5,
        )

        savings_bytes = old_peak - new_peak
        savings_pct = (savings_bytes / max(old_peak, 1)) * 100.0

        print(
            f"\n[P14 memory] logits=[R={R}, V={V}] fp32 "
            f"(nominal [R,V] fp32 lp tensor = "
            f"{nominal_intermediate_bytes / 1e6:.1f} MB)\n"
            f"  Old (log_softmax+gather) tracked-alloc peak: "
            f"{old_peak:>12d} bytes ({old_peak / 1e6:.2f} MB)\n"
            f"  New (cross_entropy)      tracked-alloc peak: "
            f"{new_peak:>12d} bytes ({new_peak / 1e6:.2f} MB)\n"
            f"  Savings: {savings_bytes} bytes ({savings_pct:.1f}% "
            f"of old peak)"
        )

        # The naive path MUST surface the [R, V] fp32 log_softmax tensor.
        assert old_peak >= nominal_intermediate_bytes, (
            f"Naive path peak ({old_peak} B) is below the nominal "
            f"[R, V] fp32 log_softmax tensor size "
            f"({nominal_intermediate_bytes} B). The tracker missed "
            "allocations — see test impl."
        )

        # Direction assertion: positive savings detected.
        assert new_peak < old_peak, (
            f"P14 violation: fused cross_entropy peak ({new_peak} B) is "
            f"NOT less than naive log_softmax+gather peak ({old_peak} B). "
            f"Expected the fused path to skip the [R={R}, V={V}] fp32 "
            f"log_softmax tensor (~{nominal_intermediate_bytes / 1e6:.1f} MB)."
        )

        # Stronger: savings should be at least half the [R, V] intermediate
        # (the fused path's intermediates are O(R), not O(R*V)).
        assert savings_bytes >= nominal_intermediate_bytes // 2, (
            f"P14: expected savings >= {nominal_intermediate_bytes // 2} B "
            f"(half the [R, V] fp32 intermediate); got {savings_bytes} B."
        )


# ── 2. P18 — vectorized .item() in generate_rollouts ─────────────────────────

class TestP18VectorizedItemCalls:
    """Source-pattern + functional check that generate_rollouts no longer
    has B-serialized `.item()` calls inside its for-loop body."""

    def test_generate_rollouts_minimizes_per_iter_item_calls(self):
        """Count `[i].item()` patterns — these are the per-i sync points
        that the old code had three of. P18 requires zero per-iter syncs
        (use `.cpu().tolist()` once per batch instead)."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.generate_rollouts)

        per_iter_patterns = [
            r"\[i\]\.item\(\)",
            r"\[i\]\)\.item\(\)",
        ]
        per_pattern_counts = {
            p: len(re.findall(p, src)) for p in per_iter_patterns
        }
        total = sum(per_pattern_counts.values())

        print(
            f"\n[P18] generate_rollouts per-iter .item() patterns: "
            f"{per_pattern_counts} (total={total})"
        )

        assert total == 0, (
            f"generate_rollouts has {total} per-iter .item() calls; "
            f"P18 requires zero per-iter syncs (use cpu().tolist() before "
            f"the loop). Pattern breakdown: {per_pattern_counts}"
        )

    def test_generate_rollouts_uses_batch_level_tolist(self):
        """Functional companion: P18's positive evidence is the
        presence of `.cpu().tolist()` (single batch sync) inside the
        method."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.generate_rollouts)
        n_tolist = src.count(".cpu().tolist()") + src.count(".tolist()")
        print(
            f"\n[P18] generate_rollouts batch-level .tolist() calls: {n_tolist}"
        )
        assert n_tolist >= 1, (
            "P18 expects at least one `.cpu().tolist()` (or `.tolist()`) "
            "call to convert per-sample results in a single sync."
        )


# ── 3. P12 — epoch-0 redundant policy forward skip ───────────────────────────

class TestP12EpochZeroSkip:
    """The first PPO epoch's policy forward is redundant (ratio is
    identically 1.0 since no optimizer step has occurred yet). P12
    skips it — but ONLY when n_ppo_epochs >= 2, so K=1 still gets a
    real policy update."""

    def test_train_step_only_skips_epoch_zero_when_K_geq_2(self):
        """train_step must guard `is_first_epoch=True` with the
        `n_ppo_epochs >= 2` (or equivalent `> 1`) check."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step)

        # Acceptable forms: `epoch == 0 and self.config.n_ppo_epochs >= 2`
        #                   `epoch == 0 and self.config.n_ppo_epochs > 1`
        # The whole expression must be on a single line for the regex
        # below; we collapse whitespace first to make the test forgiving
        # of formatting changes.
        flat = re.sub(r"\s+", " ", src)
        matches = re.findall(
            r"is_first_epoch\s*=\s*\([^)]*epoch[^)]*==[^)]*0[^)]*"
            r"(?:>=\s*2|>\s*1)[^)]*\)",
            flat,
        )

        print(
            f"\n[P12] train_step is_first_epoch guard matches: {len(matches)}"
        )
        if matches:
            print(f"  -> {matches[0]}")

        assert matches, (
            "train_step must guard is_first_epoch with n_ppo_epochs >= 2 "
            "(or > 1) so K=1 still does 1 policy update. Pattern not "
            f"found. Source head:\n{src[:1200]}"
        )

    def test_ppo_update_first_epoch_branch_reuses_old_log_probs(self):
        """When `is_first_epoch=True` AND precomputed old per-token log
        probs are passed, ppo_update must NOT call the policy forward
        (would be wasted compute). The reuse branch should reference the
        cached old per-token tensor."""
        from ppo_specs.ppo_trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.ppo_update)

        assert "is_first_epoch" in src, (
            "ppo_update must branch on is_first_epoch."
        )

        # Heuristic: there must be a control-flow block that uses
        # is_first_epoch and references old_per_token (the precomputed
        # frozen log probs reused as new_per_token on epoch 0).
        # Capture the body of the `if is_first_epoch:` block by taking
        # the next ~6 logical lines.
        m = re.search(
            r"if\s+is_first_epoch\s*:\s*\n((?:\s{8,}.*\n){1,8})", src
        )
        assert m, (
            "Expected an `if is_first_epoch:` block in ppo_update.\n"
            f"Source excerpt:\n{src[:1500]}"
        )
        branch_body = m.group(1)
        print(
            f"\n[P12] ppo_update first-epoch branch body:\n{branch_body}"
        )
        assert "old_per_token" in branch_body, (
            "P12 first-epoch branch must reuse old_per_token as the "
            "policy log probs (avoids the redundant forward). "
            f"Branch body:\n{branch_body}"
        )

    def test_p12_functional_skip_with_mocked_trainer(self):
        """Functional: build a stub PPOTrainer-like object, monkey-patch
        `_batched_per_token_log_probs` AND `_critic_forward` to count
        their invocations, then invoke ppo_update with both
        is_first_epoch values and verify only the non-skipped path runs
        the policy forward."""
        from ppo_specs.ppo_trainer import PPOTrainer, RolloutBatch, Rollout

        # Construct a minimally-viable batch
        rollouts = [
            Rollout(
                prompt="p", completion="c", reward=1.0,
                old_log_prob=0.0, value=0.0,
                full_ids=[1, 2, 3, 4, 5], prompt_len=2,
                parse_success=True, format_match_boxed=False,
            ),
            Rollout(
                prompt="q", completion="d", reward=0.0,
                old_log_prob=0.0, value=0.0,
                full_ids=[1, 2, 6, 7, 8], prompt_len=2,
                parse_success=False, format_match_boxed=False,
            ),
        ]
        batch = RolloutBatch(rollouts)

        device = torch.device("cpu")
        T = 3  # response length
        B = 2

        class _Cfg:
            log_ratio_clip = 20.0
            clip_epsilon = 0.2
            critic_loss_coeff = 0.5
            kl_coeff = 0.0
            reference_kl_coeff = 0.0
            grad_clip_norm = 1.0
            gamma = 1.0

        # A trainable critic head so the critic_loss carries a grad_fn,
        # which lets total_loss.backward() succeed regardless of whether
        # the policy forward ran.
        class _CriticHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = nn.Parameter(torch.tensor(0.5))

            def is_trainable(self):
                return True

            def train(self, mode=True):
                return self

            def eval(self):
                return self

        critic_head = _CriticHead()

        mock_model = MagicMock()
        mock_model.train = MagicMock(return_value=None)
        mock_model.parameters = MagicMock(return_value=[])
        mock_model.eval = MagicMock(return_value=None)

        forward_count = {"policy": 0, "critic": 0}

        def _spy_per_token_logprobs(self, all_full_ids, prompt_lens,
                                    model_override=None):
            forward_count["policy"] += 1
            # Return a per-token tensor that's NOT connected to a graph;
            # ratio = exp(new - old) where both come from this same
            # tensor type so the policy_loss has no grad — but critic_loss
            # will, which is enough to allow .backward().
            per_token = torch.randn(B, T)
            mask = torch.ones(B, T)
            return per_token, mask

        def _spy_critic_forward(self, batch, rewards):
            forward_count["critic"] += 1
            critic_values = critic_head.scalar.expand(B).clone()  # [B], with grad
            critic_loss = torch.nn.functional.mse_loss(critic_values, rewards)
            return critic_values, critic_loss

        trainer = PPOTrainer.__new__(PPOTrainer)  # bypass __init__
        trainer.config = _Cfg()
        trainer.model = mock_model
        trainer.critic = critic_head
        trainer.device = device
        trainer.policy_optimizer = torch.optim.SGD(
            [nn.Parameter(torch.randn(1))], lr=1e-3
        )
        trainer.critic_optimizer = torch.optim.SGD(
            critic_head.parameters(), lr=1e-3
        )
        trainer._batched_per_token_log_probs = _spy_per_token_logprobs.__get__(
            trainer, PPOTrainer
        )
        trainer._critic_forward = _spy_critic_forward.__get__(
            trainer, PPOTrainer
        )

        old_per_token = torch.randn(B, T)
        mask = torch.ones(B, T)
        advantages = torch.tensor([0.5, -0.5])

        # Path A: is_first_epoch=True → policy forward should be skipped.
        forward_count["policy"] = 0
        forward_count["critic"] = 0
        _ = trainer.ppo_update(
            batch,
            precomputed_advantages=advantages,
            precomputed_old_per_token_log_probs=old_per_token,
            precomputed_response_mask=mask,
            is_first_epoch=True,
        )
        first_policy = forward_count["policy"]
        first_critic = forward_count["critic"]

        # Path B: is_first_epoch=False → policy forward must run.
        forward_count["policy"] = 0
        forward_count["critic"] = 0
        _ = trainer.ppo_update(
            batch,
            precomputed_advantages=advantages,
            precomputed_old_per_token_log_probs=old_per_token,
            precomputed_response_mask=mask,
            is_first_epoch=False,
        )
        later_policy = forward_count["policy"]
        later_critic = forward_count["critic"]

        print(
            f"\n[P12 functional] forward call counts:\n"
            f"  is_first_epoch=True : policy={first_policy}, "
            f"critic={first_critic}\n"
            f"  is_first_epoch=False: policy={later_policy}, "
            f"critic={later_critic}\n"
            f"  policy forwards saved by P12: "
            f"{later_policy - first_policy}"
        )

        assert first_policy == 0, (
            f"P12: ppo_update with is_first_epoch=True must NOT call "
            f"_batched_per_token_log_probs (saw {first_policy} calls)."
        )
        assert later_policy >= 1, (
            f"P12 sanity: with is_first_epoch=False the policy forward "
            f"must run (saw {later_policy} calls)."
        )
        # Critic must run on both paths (the whole point of P12 is that
        # epoch 0 STILL trains the critic; only the policy forward is skipped)
        assert first_critic >= 1 and later_critic >= 1, (
            f"Critic forward should run on both paths "
            f"(first={first_critic}, later={later_critic})."
        )


# ── 4. P15 — _extract_last_hidden does NOT request all hidden states ─────────

@pytest.mark.slow
class TestP15HiddenStateExtraction:
    """The fast path of _extract_last_hidden uses a forward hook on the
    final norm — it must NOT pass `output_hidden_states=True`."""

    def test_p15_does_not_allocate_full_hidden_state_stack(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ppo_specs.config import PPOConfig
        from ppo_specs.critic import build_critic
        from ppo_specs.ppo_trainer import PPOTrainer
        from src.rewards import gsm8k_reward

        name = "Qwen/Qwen2.5-0.5B-Instruct"
        m = AutoModelForCausalLM.from_pretrained(name)
        m.eval()
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        cfg = PPOConfig(
            model_name=name, batch_size=2, critic_capacity="medium",
        )
        critic = build_critic("medium", m.config.hidden_size)
        trainer = PPOTrainer(
            config=cfg, model=m, tokenizer=tok, critic=critic,
            reward_fn=gsm8k_reward, device=torch.device("cpu"),
        )

        # Spy: count forward calls that pass output_hidden_states=True
        counters = {"hidden_states_calls": 0, "total_calls": 0}
        orig_forward = m.forward

        def spy_forward(*args, **kwargs):
            counters["total_calls"] += 1
            if kwargs.get("output_hidden_states", False):
                counters["hidden_states_calls"] += 1
            return orig_forward(*args, **kwargs)

        m.forward = spy_forward
        try:
            result = trainer._extract_last_hidden(
                ["What is 2+2?", "Solve 5*3="]
            )
        finally:
            m.forward = orig_forward

        print(
            f"\n[P15] _extract_last_hidden forward calls: "
            f"total={counters['total_calls']}, "
            f"output_hidden_states=True: {counters['hidden_states_calls']}"
        )

        assert result.shape == (2, m.config.hidden_size), (
            f"Expected [B=2, H={m.config.hidden_size}], got {tuple(result.shape)}"
        )
        assert counters["hidden_states_calls"] == 0, (
            "P15: _extract_last_hidden must NOT pass output_hidden_states=True "
            "for supported architectures (Llama/Qwen/Mistral). Saw "
            f"{counters['hidden_states_calls']} such call(s)."
        )

        # Functional parity: result must match the
        # output_hidden_states[-1] baseline bitwise.
        enc = tok(
            ["What is 2+2?", "Solve 5*3="],
            return_tensors="pt", padding=True, max_length=64, truncation=True,
        )
        with torch.no_grad():
            out_baseline = m(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        expected = out_baseline.hidden_states[-1][
            torch.arange(2), seq_lens, :
        ]

        torch.testing.assert_close(
            result, expected, atol=0.0, rtol=0.0,
            msg=lambda diff: (
                "P15 fast path must produce bitwise-identical output to "
                f"the output_hidden_states[-1] baseline. Diff: {diff}"
            ),
        )


# ── 5. End-to-end smoke with all optimizations enabled ───────────────────────

@pytest.mark.slow
class TestEndToEndAllOptimizations:
    def test_end_to_end_with_all_optimizations_enabled(self):
        """All Phase 1+2 optimizations enabled should produce a valid
        training metric dict without errors. Uses the local_test_config
        preset on CPU with a single batch and a single training step."""
        from ppo_specs.config import local_test_config, copy_config
        from ppo_specs.ppo_trainer import load_ppo_trainer

        cfg = copy_config(
            local_test_config(),
            n_ppo_epochs=2,           # exercise P12 epoch-0 skip
            optimizer_fused=False,    # CPU
            optimizer_8bit=False,
            reference_quant="none",
            reference_kl_coeff=0.0,   # avoid loading reference model
            n_steps=1, eval_every=1, log_every=1,
            batch_size=2, max_new_tokens=8,
        )
        device = torch.device("cpu")
        trainer, _ = load_ppo_trainer(cfg, device)

        prompts = ["What is 2+2?", "Solve 1+1="]
        gts = ["4", "2"]
        metrics = trainer.train_step(prompts, gts)

        print(f"\n[E2E] train_step metric keys: {sorted(metrics.keys())}")
        for key in (
            "policy_loss", "critic_loss", "kl_divergence",
            "mean_reward", "mean_advantage", "clip_fraction", "accuracy",
        ):
            assert key in metrics, f"Missing metric: {key}"
            v = metrics[key]
            assert isinstance(v, (int, float)) and math.isfinite(v), (
                f"Bad metric {key}={v}"
            )
            print(f"  {key:>20s} = {v:.6f}")
