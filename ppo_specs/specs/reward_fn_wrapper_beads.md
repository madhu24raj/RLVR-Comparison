# `_RewardFnWrapper` Batched-Path Beads

Companion to [reward_model_integration.md](reward_model_integration.md) and
[integration_beads.md](integration_beads.md). The Phase 1 fix in
[reward_model_integration.md §"Performance gap in `SelfJudgeRewardModel`"](reward_model_integration.md#performance-gap-in-selfjudgerewardmodel-added-2026-04-30-deep-review)
landed `SelfJudgeRewardModel.batch_score` (one fused forward instead of B
single-sample forwards). The actual perf win — ~5 minutes per 200-step run
at 8B in self_judge mode — only materializes when callers stop going through
the per-sample `__call__` path. This file defines the beads to wire that up.

## Status snapshot (2026-05-04)

| Component | Status |
|-----------|--------|
| `SelfJudgeRewardModel.batch_score` fused implementation | **DONE** ([src/rewards.py:195-304](../../src/rewards.py#L195-L304)) |
| `_RewardFnWrapper.batch_call` batched method | **NOT STARTED** (this file's RFW-A) |
| Trainer batched-path detection in `generate_rollouts` / `evaluate` | **NOT STARTED** (RFW-B) |
| Unit tests for `batch_call` parity vs `__call__` | **NOT STARTED** (RFW-TEST) |
| Smoke verification under DDP | **NOT STARTED** (RFW-VERIFY) |

## Baseline invariant (locked, applies to every bead)

> When `reward_mode == "deterministic"` (the default), the per-sample
> `self.reward_fn(completion, gt)` call site stays bit-identical to today.
> The new `batch_call` is an *opt-in fast path* that the trainer prefers
> when the wrapper exposes it, but the legacy path remains the canonical
> implementation for any reward_fn that doesn't carry the new method.

Same baseline shape as the learned-RM `RewardModelScorer` integration:
batched call when available, per-sample fallback otherwise. The two
batched paths (this one for self-judge / combined modes; the
`RewardModelScorer.score_batch` path for the learned RM) are mutually
exclusive — `load_ppo_trainer` already raises `ValueError` when both
would fire (per [reward_model_integration.md §"Interaction with
reward_mode"](reward_model_integration.md#interaction-with-reward_mode-orthogonality-contract)).

## Dispatch order

```
RFW-A  ──►  RFW-B  ──►  RFW-VERIFY
    └──►  RFW-TEST  ───────┘
```

RFW-B and RFW-TEST can run in parallel after RFW-A. RFW-VERIFY runs last
on the merged branch.

---

## RFW-A — Add `batch_call` to `_RewardFnWrapper`

**Owner:** `general-purpose` sub-agent. **Estimated time:** ~45 min.

**What this delivers:** a `batch_call(completions, ground_truths, questions=None) -> list[float]` method on `_RewardFnWrapper` that handles all three `reward_mode` values via one fused call to `SelfJudgeRewardModel.batch_score` (when applicable) instead of B per-sample forwards.

**Files to read:**
- [src/rewards.py:131-198](../../src/rewards.py#L131-L198) — `SelfJudgeRewardModel`. Note the new `batch_score(questions, completions) -> list[float]` at [L195-L304](../../src/rewards.py#L195-L304).
- [src/rewards.py:201-285](../../src/rewards.py#L201-L285) — `_RewardFnWrapper` (existing per-sample `__call__`, `set_questions`, `_idx`) and `make_reward_fn` factory. The wrapper is stateful: `set_questions(questions)` is called from the trainer before each batch, and `__call__(completion, gt)` reads the i'th question via the `_idx` counter.
- [src/rewards.py:84-105](../../src/rewards.py#L84-L105) — `gsm8k_reward(completion, gt) -> float` (the deterministic verifier).
- [reward_model_integration.md §"Adapter contract"](reward_model_integration.md#adapter-contract) — the design pattern your batched method should mirror.

**Files to edit (one file):** `src/rewards.py`.

**Method to add** on `_RewardFnWrapper` (next to the existing `__call__`):

```python
def batch_call(
    self,
    completions: list[str],
    ground_truths: list[str],
    questions: list[str] | None = None,
) -> list[float]:
    """Batched analog of __call__: one fused forward for the self-judge
    component instead of B per-sample forwards.

    Args:
        completions:    list[str], one per sample.
        ground_truths:  list[str], one per sample. Used for the
                        deterministic component (gsm8k_reward).
        questions:      list[str] | None. Required when reward_mode is
                        "self_judge" or "combined". Pass the prompt /
                        question that produced each completion. If None
                        and the wrapper has internal state from a prior
                        set_questions(...) call, use that instead — but
                        the explicit argument is preferred.

    Returns:
        list[float] of length len(completions), aligned positionally.

    Bit-identical to [self(c, gt) for c, gt in zip(completions, ground_truths)]
    in deterministic mode and within numerical noise (fp32 vs per-sample
    fp32) for self_judge / combined modes. The `__call__` path remains
    available and unchanged.
    """
```

Implementation logic:
- `reward_mode == "deterministic"`: iterate `gsm8k_reward(c, gt)` per sample. (No self-judge component → no perf win possible; this branch exists for callers that prefer the uniform batch interface.)
- `reward_mode == "self_judge"`: resolve `questions` (parameter, then `self._questions`, then raise `ValueError`), call `self.judge.batch_score(questions, completions)` once, return the resulting list. Honor the `self_judge_normalize` flag — already baked into `batch_score` via `self.normalize`.
- `reward_mode == "combined"`: compute both — `det = [gsm8k_reward(c, gt) for c, gt in ...]` and `judge = self.judge.batch_score(questions, completions)` — then blend by `self_judge_weight`: `out[i] = (1 - w) * det[i] + w * judge[i]`. Match the exact blend direction of the per-sample `__call__` (verify by reading L227-239 of the existing wrapper).

**Stateful counter handling:** `batch_call` should NOT touch `self._idx`. The counter is for the per-sample path only. Document that the two paths are mutually exclusive within a single batch — the trainer picks one or the other but never both.

**Invariants to preserve:**
- `__call__` is unchanged. Existing per-sample callers (today's trainer, any test) keep working.
- `set_questions(...)` semantics unchanged.
- `make_reward_fn(...)` factory return signature unchanged: still returns `(reward_fn, diagnostic_fn)`. The new method is on the existing wrapper; no API surface change.

**Hard preconditions:** none. RFW-A can land before any other bead.

**Self-verification:**
```bash
python -c "
from src.rewards import _RewardFnWrapper, gsm8k_reward
# Smoke: build a wrapper for deterministic mode (no real model needed),
# verify batch_call on a 3-item batch matches per-sample __call__.
import types
cfg = types.SimpleNamespace(
    reward_mode='deterministic', self_judge_weight=0.5, self_judge_normalize=False,
)
w = _RewardFnWrapper(cfg, judge=None)
gts = ['42', '25', '100']
comps = [r'... \boxed{42}', '#### 25', 'no answer']
batched = w.batch_call(comps, gts)
serial = [gsm8k_reward(c, gt) for c, gt in zip(comps, gts)]
assert batched == serial, f'parity broken: batch={batched} serial={serial}'
print('deterministic parity ok:', batched)
"
```
And by inspection: `grep -n "batch_call" src/rewards.py` shows the new method.

**Reporting contract:** the file touched, a one-line confirmation that deterministic-mode parity holds, and the exact blend formula used for combined mode (paste the line so the reviewer can verify it matches `__call__`).

---

## RFW-B — Wire trainer to prefer `batch_call` when available

**Owner:** `general-purpose` sub-agent. **Estimated time:** ~45 min.

**What this delivers:** the trainer's `generate_rollouts` and `evaluate`
detect `hasattr(self.reward_fn, 'batch_call')`. When true, they call the
batched method once per batch instead of looping per-sample. When false,
the existing per-sample loop runs (preserves baseline invariant for any
non-wrapper `reward_fn` callable).

**Files to read:**
- `src/rewards.py` (after RFW-A lands) — the new `batch_call` method.
- [ppo_specs/specs/reward_model_integration.md "Adapter contract"](reward_model_integration.md#adapter-contract) — the matching pattern already used for `RewardModelScorer.score_batch`.
- [ppo_specs/ppo_trainer.py:267-389](../ppo_trainer.py#L267-L389) — `generate_rollouts`. The per-sample reward call is at ~L389-L392 (the same site that already has the `RewardModelScorer` fast-path branch). The new `batch_call` branch is a sibling fast-path: if the scorer is None AND the wrapper has `batch_call`, prefer that.
- [ppo_specs/ppo_trainer.py:1041-L1124](../ppo_trainer.py#L1041-L1124) — `evaluate`. Mirror the substitution.
- [ppo_specs/run_e2_7.py](../run_e2_7.py) and [ppo_specs/run_e2_8.py](../run_e2_8.py) — the call sites that swap `trainer.reward_fn` for the diagnostic_fn during eval. Confirm `diagnostic_fn` (returned from `make_reward_fn`) does NOT need `batch_call` — it should remain the per-sample gsm8k_reward path because it's used purely for the binary accuracy metric.

**Files to edit (one file):** `ppo_specs/ppo_trainer.py`.

### Edit 1: `generate_rollouts` (~L370-L400)

The existing structure (after RM-B's earlier edit):
```python
rm_scores_list: Optional[List[float]] = None
if self.reward_model_scorer is not None:
    rm_scores = self.reward_model_scorer.score_batch(
        local_prompts, completions, local_gts,
    )
    rm_scores_list = rm_scores.detach().cpu().tolist()
```

Add a sibling `wrapper_batch_list` branch immediately after, gated on
`hasattr`:

```python
wrapper_batch_list: Optional[List[float]] = None
if rm_scores_list is None and hasattr(self.reward_fn, "batch_call"):
    wrapper_batch_list = self.reward_fn.batch_call(
        completions=completions,
        ground_truths=local_gts,
        questions=local_prompts,
    )
```

In the per-sample loop that constructs each `Rollout`, the existing
priority chain becomes:
```python
if rm_scores_list is not None:
    reward = float(rm_scores_list[i])
elif wrapper_batch_list is not None:
    reward = float(wrapper_batch_list[i])
else:
    reward = self.reward_fn(completion, local_gts[i])
```

The `det_reward = gsm8k_reward(completion, local_gts[i])` line at L401
stays unchanged — accuracy still always reduces over the binary verifier.

### Edit 2: `evaluate` (~L1063-L1120)

Same shape: detect `batch_call`, call once on the local shard, populate
the per-sample reward list from the returned floats. The accuracy metric
already comes from `gsm8k_reward(c, gt)` regardless of which path
populated `r.reward` (existing invariant), so no further change.

### Edit 3 (optional polish): a fast-path detection log line

In `load_ppo_trainer`, after the wrapper is built (after the
`make_reward_fn` call at ~L1268-L1271), add a one-line `_print0` so the
operator sees which reward path is active:
```python
if hasattr(reward_fn, "batch_call"):
    _print0(f"[PPO] reward_fn supports batch_call ({config.reward_mode}); "
            f"using batched fast path")
```
Optional — useful for debugging but not load-bearing.

**Hard preconditions:** RFW-A (provides `batch_call`).

**Invariants to preserve:**
- Baseline: when `reward_mode == "deterministic"` AND no learned RM is
  configured, the trainer behavior must be bit-identical to today's
  per-sample loop. `batch_call` for deterministic mode IS bit-identical
  per RFW-A's invariant, but if RFW-A's tests show fp32 noise in this
  branch, gate the fast path on `reward_mode != "deterministic"` instead
  of `hasattr` to keep the baseline trivially preserved.
- The `RewardModelScorer.score_batch` path takes precedence over
  `batch_call` (already the order in the priority chain above).
- The diagnostic eval path (which swaps `trainer.reward_fn` for
  `diagnostic_fn`) continues to use the per-sample loop because
  `diagnostic_fn` does NOT carry `batch_call`. Verify this is true after
  RFW-A lands.

**Self-verification:**
```bash
# Baseline parity (deterministic mode → bit-identical to pre-RFW)
PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc 2>&1 | tail -10

# Fast-path engages in self_judge mode (the actual perf win)
PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc --reward-mode self_judge 2>&1 | grep "batch_call\|step"

# grep confirmations
grep -n "batch_call" ppo_specs/ppo_trainer.py    # ≥3 hits (init scan, generate_rollouts, evaluate)
grep -n "self.reward_fn(" ppo_specs/ppo_trainer.py  # ≥2 hits (the per-sample fallbacks must remain)
```

**Reporting contract:** the line ranges touched, the deterministic-mode
parity result (paste last 5 lines of the smoke output), and a one-line
note on whether the optional log line in `load_ppo_trainer` was added.

---

## RFW-TEST — Unit tests for `batch_call` parity

**Owner:** `general-purpose` sub-agent. **Estimated time:** ~30 min.

**What this delivers:** `ppo_specs/tests/test_reward_fn_wrapper.py` with
4 tests that lock in `batch_call` correctness across all three reward
modes.

**Files to read:**
- `src/rewards.py` (after RFW-A lands) — the new `batch_call`.
- [ppo_specs/tests/test_data_rewards.py](../tests/test_data_rewards.py) — for the existing reward-test class style and assertion idioms.
- [ppo_specs/tests/test_reward_model.py](../tests/test_reward_model.py) — for the monkeypatch pattern (mocked `AutoModelForCausalLM`).

**Files to edit:**
- **NEW** `ppo_specs/tests/test_reward_fn_wrapper.py`. Match
  `test_data_rewards.py`'s class layout. Six tests:

1. `test_deterministic_parity` — `batch_call` output is `==` (float-exact) to per-sample `__call__` for a 3-item hand-rolled batch in deterministic mode. No mock needed.

2. `test_self_judge_parity` — mock the judge so `batch_score` and `score` both return a known constant function of input length. Confirm `batch_call` output matches per-sample `__call__` output element-wise (within `1e-6` for fp32 noise).

3. `test_combined_blend_direction` — for `self_judge_weight ∈ {0, 0.5, 1}`, confirm: w=0 reproduces deterministic mode, w=1 reproduces self_judge mode, w=0.5 is the element-wise convex combination. Tolerance `1e-6`.

4. `test_questions_required_for_self_judge` — `batch_call(..., questions=None)` raises `ValueError` (or falls back to `self._questions` if set, then raises if that's also unset). Match the path your RFW-A implementation took.

5. `test_idx_counter_untouched` — `batch_call` does NOT increment `self._idx`. Set `self._idx = 7` before, run `batch_call`, assert `self._idx == 7` after.

6. `test_empty_batch` — `batch_call([], [], [])` returns `[]` cleanly, doesn't raise, doesn't trigger the model.

**Hard preconditions:** RFW-A.

**Invariants to preserve:**
- CPU-only. No `torch.cuda.*`.
- No network. Mock the HF model in `SelfJudgeRewardModel`.
- Total runtime < 5 s.

**Self-verification:**
```bash
pytest ppo_specs/tests/test_reward_fn_wrapper.py -v
pytest ppo_specs/tests/ -v   # no collateral damage
```

**Reporting contract:** test count, total runtime, and any test that had
to be adapted from the spec's plan with a one-line reason.

---

## RFW-VERIFY — Smoke + perf check on the merged branch

**Owner:** `general-purpose` sub-agent. **Estimated time:** ~30 min.

**What this delivers:** confirmation that the batched fast-path engages
end-to-end and produces correct outputs on the local smoke. Optionally
times before/after to validate the perf win on a small model (the 8B
win is the real prize, but a 0.5B baseline shows the path fires).

**Files to read:**
- [ppo_specs/specs/reward_model_integration.md §"Performance gap"](reward_model_integration.md#performance-gap-in-selfjudgerewardmodel-added-2026-04-30-deep-review).
- The current state of `ppo_specs/ppo_trainer.py` after RFW-B.

**Files to edit:** none (verification only).

**Hard preconditions:** RFW-A, RFW-B, RFW-TEST all landed.

**Steps:**

1. Deterministic-mode smoke (baseline parity):
   ```bash
   PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc
   ```
   Final accuracy, per-step `mean_reward`/`policy_loss`/`kl_divergence` must match a known-good pre-RFW run (run a copy on the parent commit if needed, save the JSON, diff). Bit-identical or within numerical noise expected.

2. Self-judge mode smoke (the path the bead exists for):
   ```bash
   PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc \
       --reward-mode self_judge
   ```
   Should complete 5 steps. The `[PPO] reward_fn supports batch_call` line (if RFW-B added it) should appear once. No "B times" repeated debug output.

3. Combined-mode smoke:
   ```bash
   PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc \
       --reward-mode combined
   ```
   Same shape — completes, batched fast-path engages.

4. Multi-process CPU DDP under self_judge:
   ```bash
   PYTHONUTF8=1 accelerate launch \
       --config_file configs/accelerate_cpu.yaml \
       ppo_specs/run_e2_7.py --local-test --no-mc --reward-mode self_judge
   ```
   No deadlock. `batch_call` is called once per rank on the local shard.
   `set_questions` is also called per local-shard (existing §7.6.2 fix).
   Confirm gathered rewards are consistent across ranks.

5. Timing comparison (optional, on whatever GPU is available):
   ```bash
   # Before RFW (parent commit, --reward-mode self_judge): record total step time.
   # After RFW: same command. Expect modest (~2-5x) speedup on 0.5B; the
   # 8B speedup is ~5 min/200-step run as projected in the spec.
   ```

**Reporting contract:** the table from steps 1-4 (mode × wall time × final_acc × any errors), the timing comparison if step 5 was run, and any unexpected behavior.

---

## Notes for the dispatcher

- **Worktrees recommended.** RFW-A and RFW-TEST touch different files
  (`src/rewards.py` vs `ppo_specs/tests/test_reward_fn_wrapper.py`); RFW-B
  touches a third (`ppo_specs/ppo_trainer.py`). Three separate worktrees
  let A → (B ‖ TEST) parallelize cleanly.
- **Don't dispatch RFW-VERIFY before RFW-TEST passes.** The smoke runs
  are the integration check; the unit tests are the contract check.
  Skipping the contract check makes integration failures harder to
  triage.
- **Do not modify `RewardModelScorer.score_batch`** (the learned-RM
  path). It is orthogonal — already wired and tested. The two batched
  paths are mutually exclusive at runtime per the orthogonality contract.
- **Bit-identical baseline check.** If RFW-A's deterministic-mode
  parity test (RFW-TEST #1) finds even fp32-level drift, RFW-B should
  gate the fast path on `reward_mode != "deterministic"` rather than
  `hasattr` to keep the baseline trivially preserved. The deterministic
  branch in `batch_call` is just a `gsm8k_reward` loop, so this should
  not happen — flag if it does.
- **8B perf win.** The realized speedup is ~1.4 s/step at B=16 on 8B
  per the spec, ≈5 minutes per 200-step E2.7 run in self_judge or
  combined mode. Negligible for deterministic-only runs. Document the
  numbers in the verification report.
