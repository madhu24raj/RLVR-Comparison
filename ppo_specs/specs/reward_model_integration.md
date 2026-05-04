# Learned Reward Model Integration for PPO

Status: DESIGN
Author: planning agents
Date: 2026-04-24

## Context

Today the PPO pipeline uses a **verifiable** reward (`gsm8k_reward` in [src/rewards.py:80-101](../../src/rewards.py#L80-L101)) — a deterministic parser that extracts `\boxed{...}` answers and checks equality against ground truth. This is the "V" in RLVR.

We want to add a **learned reward model (RM)** — a pretrained causal LM fine-tuned to score (prompt, completion) pairs — as an alternative or complementary reward signal. Final target: 8B-parameter RM running on the GPU cluster. Intermediate target: CPU-smoke-test at a 0.5B RM tier so distributed bugs don't burn GPU allocations.

This spec defines the integration surface, memory budget, capacity tiers, and testing strategy. It assumes the pending Accelerate-integration work (see [distributed.md](distributed.md) and the `PPO Cluster DDP Enablement` plan) is in place or proceeds in parallel — the RM must work with single-process CPU, multi-process CPU DDP, and multi-GPU DDP with no code path divergence.

## Current implementation status (2026-04-30)

A **partial** reward-model layer is already in the repo, landed by the partner.
This spec describes the FULL learned-RM tier sweep that should be built ON TOP
of what's there. Take stock before starting:

| Already in repo | Location | Notes |
|-----------------|----------|-------|
| `gsm8k_reward()` deterministic verifier | [src/rewards.py:84-105](../../src/rewards.py#L84-L105) | Unchanged; still the source of truth for `accuracy`. |
| `extract_answer_from_completion`, `matches_boxed_format` | [src/rewards.py:15-68](../../src/rewards.py#L15-L68) | Used for parse/format diagnostics. |
| `SelfJudgeRewardModel` (continuous self-judge via log-likelihood) | [src/rewards.py:131-198](../../src/rewards.py#L131-L198) | Frozen reference model scoring; not the same as a trained RM. |
| `_RewardFnWrapper` (stateful wrapper, blends det + self-judge) | [src/rewards.py:201-239](../../src/rewards.py#L201-L239) | Already supports `reward_mode = "deterministic" \| "self_judge" \| "combined"`. |
| `make_reward_fn(config, reference_model, tokenizer)` factory | [src/rewards.py:242-285](../../src/rewards.py#L242-L285) | Returns `(reward_fn, diagnostic_fn)` tuple. |
| `reward_mode`, `self_judge_weight`, `self_judge_normalize`, `reference_kl_coeff` | [config.py:58-60, 46-50](../config.py) | All wired through `load_ppo_trainer` and `PPOTrainer`. |
| Diagnostic metrics (`parse_success_rate`, `format_match_rate`, `reward_nonzero_rate`, `kl_ref_divergence`) | `ppo_trainer.py` and `run_e2_*.py` | Logged every eval step. |
| `load_ppo_trainer` returns `(trainer, diagnostic_fn)` | [ppo_trainer.py:1040-L1197](../ppo_trainer.py#L1040-L1197) | Run scripts unpack accordingly. |
| `shared/per_token_loss.py` (`batched_per_token_log_probs`, `clipped_surrogate_loss`, `per_token_kl`) | [shared/per_token_loss.py](../../shared/per_token_loss.py) | Per-token PPO loss; already in use. |
| Partial `_config_hash` RM fields | [checkpoint.py:223-225](../checkpoint.py#L223-L225) | `reward_model_capacity`, `reward_blend_alpha`, `reward_score_activation` already wired via `getattr` defaults. MISSING: `reward_model_name`, `reward_model_reuse_reference` (Agent A must add). |

| MISSING (this spec's scope) | Why needed |
|-----------------------------|------------|
| `ppo_specs/reward_model.py` with `RewardModelScorer` and `build_reward_model` factory | Capacity-tier dispatch (`none`/`small`/`large`) for a TRAINED preference RM, not a self-judge log-likelihood. |
| `reward_model_capacity`, `reward_model_name`, `reward_model_dtype`, `reward_model_reuse_reference`, `reward_blend_alpha`, `reward_score_activation` config fields | Selecting a learned RM and squashing its output. |
| Batched `score_batch(prompts, completions, gts)` integration in `generate_rollouts` and `evaluate` | Current code still does per-sample `self.reward_fn(c, gt)`; works for self-judge wrapper but not for a true LLM-batched RM. |
| `_config_hash` update with RM fields | Resume-safety across reward modes. |
| `tests/test_reward_model.py` | Tier parity, mock-RM shapes, blend interpolation, reuse-reference weight sharing. |
| `run_e2_rm.py` (optional sweep script) | E2.8-style sweep over RM capacity tiers. |

In short: **self-judge mode is production-ready; the tier-based learned-RM
module is not yet built**. Agents working on this spec start from the
"MISSING" list.

### Performance gap in `SelfJudgeRewardModel` (added 2026-04-30 deep review)

Although the self-judge mode is *correct* and *integrated*, it is
**not performant**. Two issues quantified:

1. **Per-sample sequential forwards.** `_RewardFnWrapper.__call__` is invoked
   B times per `train_step` from inside the per-i loop in
   [ppo_trainer.py:343](../ppo_trainer.py#L343) (inside `generate_rollouts`).
   Each call runs a single-sample forward through the reference model
   ([src/rewards.py:169](../../src/rewards.py#L169)). At 8B in self_judge
   mode: B=16 × 92 ms = **1.5 s/step** of pure single-sample reference
   inference.

2. **Doubly-redundant reference-model forwards.** The reference model is used
   for BOTH the self-judge reward AND the `kl_ref_per_token` computation
   in `train_step` ([ppo_trainer.py:903-909](../ppo_trainer.py#L903-L909)).
   These are two distinct call sites with the same input distribution
   but no sharing. Both run a full LM forward over `[prompt+completion]`.

**Combined cost at 8B, B=16:** self-judge (1.5 s) + kl_ref forward (0.35 s)
= **1.85 s of redundant ref-model work per train_step**. Over 200 steps:
6.2 minutes of pure waste in self_judge mode.

**Fix (Phase 1):** Add `SelfJudgeRewardModel.batch_score(questions, completions, prompt_lens) -> Tensor[B]`
that does ONE batched forward pass on `[prompt+completion]` for all B
samples. Drop the single-sample `score()` from the hot path:

```python
def batch_score(self, questions, completions, prompt_lens):
    # Tokenize batch
    full_texts = [q + c for q, c in zip(questions, completions)]
    enc = self.tokenizer(
        full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=self.config.max_prompt_length + self.config.max_new_tokens,
    ).to(self.device)
    # ONE forward pass (no_grad), shape [B, S, V]
    with torch.no_grad():
        outputs = self.reference_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    # Compute mean log-prob of completion tokens per sample (vectorized)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    target = enc.input_ids[:, 1:].clone()
    target_lp = log_probs[:, :-1, :].gather(2, target.unsqueeze(-1)).squeeze(-1)
    # Mask out prompt portion (only score completion tokens)
    completion_mask = ...  # 1 for tokens past prompt_len, 0 otherwise
    score = (target_lp * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
    return score  # [B]
```

Then in `_RewardFnWrapper.__call__`, replace the per-sample loop with one
`batch_score` call after generation completes.

**Fix (Phase 2):** Fuse with `kl_ref_per_token`. The
`_batched_per_token_log_probs(..., model_override=self.reference_model)`
call at [ppo_trainer.py:903-909](../ppo_trainer.py#L903-L909) already
computes per-token log-probs for the same sequences. Add an aggregate
output `mean_log_prob_completion: Tensor[B]` to `_batched_per_token_log_probs`
(via a kwarg `return_mean_per_sample=True`) and reuse for self-judge.
Eliminates the entire B-call self-judge forward in `combined`/`self_judge`
modes.

**Combined savings at 8B:** ~1.85 s/step × 200 = **6.2 min per E2.7 run**.

### Scoring memory peak when batched

When `score_batch` is implemented as proposed, its peak memory is:

- RM weights (frozen, bf16): 16 GB
- RM logits transient `[B, S, V] bf16`: `16 × 768 × 128256 × 2 = 3.0 GB`
- log_softmax fp32 transient (if not fused via cross_entropy): `16 × 768 × 128256 × 4 = 6.0 GB`

To stay safe, use the same fused `cross_entropy(reduction='none')`
pattern as P14 in [performance.md](performance.md). With fusion, the
score_batch peak is just ~3 GB transient on top of the held weights.

## Scope

**Baseline invariant (locked decision, 2026-05-04)**

> The PPO trainer is, at its most baseline, a PPO trainer with custom reward
> functions. The existing `reward_fn` callable interface, the `reward_mode`
> config knob (`"deterministic" | "self_judge" | "combined"`), the
> `_RewardFnWrapper` stateful blender, the `make_reward_fn(config,
> reference_model, tokenizer)` factory, and `SelfJudgeRewardModel` all stay.
> The new `RewardModelScorer` is one more custom reward function that plugs
> into the same `reward_fn` slot via an adapter — it does NOT bypass or
> replace the existing layer. When `reward_model_capacity == "none"`, the
> trainer behavior is bit-identical to today's deterministic / self_judge /
> combined paths (i.e., `make_reward_fn` is the sole reward source). The
> `reward_mode` and `reward_model_capacity` knobs are orthogonal: `reward_mode`
> selects the existing pipeline (deterministic verifier, self-judge log-prob,
> or a blend of those two); `reward_model_capacity` selects the optional
> learned RM that — when enabled — wraps or replaces the resulting `reward_fn`
> via `BlendedScorer`. See §"Interaction with `reward_mode`" below.

**In scope**
- New `ppo_specs/reward_model.py` module with a `build_reward_model(capacity, ...)` factory, mirroring [ppo_specs/critic.py](../critic.py).
- Capacity tiers: `none` (fallback to `gsm8k_reward`), `small` (0.5B for CPU smoke), `large` (8B for GPU cluster).
- Wiring into [ppo_specs/ppo_trainer.py](../ppo_trainer.py) at `generate_rollouts` and `evaluate`, replacing the per-sample `reward_fn(completion, gt)` call with a batched `score_rm(prompts, completions, gts)` call on the shard.
- Config knobs on [ppo_specs/config.py](../config.py): `reward_model_capacity`, `reward_model_name`, `reward_model_dtype`, `reward_model_reuse_reference` (memory optimization).
- Memory analysis for three 8B models coexisting on one GPU.
- Unit tests (mock RM) and a CPU smoke recipe.
- A `run_e2_rm.py` sweep script in the E2.8 style (optional, see §9).

**Out of scope for this work**
- Training the RM itself. Assume a checkpoint (local path or HF hub id) is produced upstream and loaded via `AutoModelForCausalLM.from_pretrained` + value head, or `AutoModelForSequenceClassification.from_pretrained`.
- Preference-pair dataset loading (no `chosen`/`rejected` plumbing exists in the PPO path — confirmed by exploration, `ppo_specs/` has zero matches for those terms).
- Online RM retraining / active learning.
- Reward model ensembling.

## Design

### Capacity tiers

Mirror the existing critic pattern ([ppo_specs/critic.py:106-131](../critic.py#L106-L131), `build_critic`).

| Capacity | Base model | Memory (bf16 weights) | Use case |
|----------|-----------|-----------------------|----------|
| `none`   | — (uses `gsm8k_reward` directly) | 0 GB | Default, baseline, fast iteration |
| `small`  | Qwen2.5-0.5B fine-tuned as RM | ~1 GB | CPU smoke on cluster; fast GPU sanity checks |
| `large`  | Llama-3-8B fine-tuned as RM | ~16 GB | Production GPU cluster |

The factory returns a `RewardModelScorer` that exposes a single method:

```python
class RewardModelScorer(nn.Module):
    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        ground_truths: list[str] | None = None,
    ) -> torch.Tensor:  # shape [B], scalar per (prompt, completion)
        ...
```

The `ground_truths` parameter is kept in the signature for backward-compat, but preference-trained RMs ignore it. The `none` tier implementation wraps `gsm8k_reward` and iterates — preserving bitwise equivalence with today's behavior.

### Architecture choice for `small` / `large`

Use `AutoModelForCausalLM` with a scalar value head — the TRL convention. Rationale:
- Reuses the existing `AutoModelForCausalLM.from_pretrained` + `_extract_last_hidden` pattern (see [ppo_specs/ppo_trainer.py:514-571](../ppo_trainer.py#L514-L571) and `batched_per_token_log_probs` in [shared/per_token_loss.py](../../shared/per_token_loss.py)).
- Ground-truth-agnostic: score = `value_head(last_hidden_of(prompt ++ completion))`.
- Same load path as the existing reference model ([ppo_specs/ppo_trainer.py:1118-1175](../ppo_trainer.py#L1118-L1175)) — simpler memory accounting.

If the upstream RM checkpoint is a `SequenceClassification` model instead, the scorer can dispatch on the loaded class — support both but default to CausalLM+value-head.

### Config knobs (PPOConfig additions)

Add to [ppo_specs/config.py](../config.py):

```python
# Learned reward model
reward_model_capacity: str = "none"        # "none" | "small" | "large"
reward_model_name: str | None = None        # HF hub id or local path; required if capacity != "none"
reward_model_dtype: str = "auto"            # auto | bfloat16 | float32
reward_model_reuse_reference: bool = False  # if True AND rm arch==policy arch: load once, use for both
reward_blend_alpha: float = 1.0             # final_reward = alpha * rm_score + (1-alpha) * gsm8k_reward
reward_score_activation: str = "sigmoid"    # "sigmoid" | "tanh" | "none" — see "Output scale" below
```

`reward_blend_alpha` lets us mix verifiable + learned signal — `alpha=0` reproduces pre-RM behavior exactly; `alpha=1` is RM-only; intermediate values support the "verifiable reward with learned shaping" hybrid. Default `1.0` with capacity `"none"` still dispatches through the wrapper and uses `gsm8k_reward` — bitwise identical to today.

### Output scale

The raw value-head output of an `AutoModelForCausalLM + linear` is unbounded
(any real number). The deterministic verifier (`gsm8k_reward`) returns
`{0, 1}`. Mixing them via `reward_blend_alpha` therefore requires that the
RM output be squashed to a comparable range, otherwise blending at
`alpha=0.5` is dominated by whichever signal happens to have larger
magnitude.

`reward_score_activation` controls the squash applied INSIDE
`RewardModelScorer.score_batch` before the score is returned:

| Setting | Output range | When to use |
|---------|--------------|-------------|
| `"sigmoid"` (default) | `(0, 1)` | Matches `gsm8k_reward` range; safe default for blending |
| `"tanh"` | `(-1, 1)` | RM was trained with preference targets in `{-1, +1}` |
| `"none"` | `(-∞, +∞)` | RM-only training (`alpha=1`) AND advantage normalization handles scale |

The default is `"sigmoid"` because (a) it's safe under blending and
(b) advantage normalization (`(A - mean) / std`) is robust to the
narrowed range. Picking `"none"` with `alpha < 1` is a valid choice
only if the user has measured the RM's output statistics and verified
the blend doesn't collapse to one signal.

### Interaction with `reward_mode` (orthogonality contract)

`reward_mode` and `reward_model_capacity` are orthogonal axes that compose
as follows. The trainer always calls `self.reward_fn` (or `self.reward_model_scorer`,
which adapts to the same protocol — see "Adapter contract" below). The
single source of `self.reward_fn` is decided at `load_ppo_trainer` time
according to this table:

| `reward_model_capacity` | `reward_mode`     | What `reward_fn` resolves to |
|-------------------------|-------------------|------------------------------|
| `"none"`                | `"deterministic"` | `gsm8k_reward` (today's default; bit-identical) |
| `"none"`                | `"self_judge"`    | `_RewardFnWrapper(judge=SelfJudgeRewardModel(reference_model), det=None, weight=0)` (unchanged) |
| `"none"`                | `"combined"`      | `_RewardFnWrapper(..., det=gsm8k_reward, weight=self_judge_weight)` (unchanged) |
| `"small"` / `"large"`   | `"deterministic"` | `RewardModelScorer.score_batch` adapted to per-sample callable (RM is the sole training reward) |
| `"small"` / `"large"`   | `"self_judge"`    | RESERVED — see "Conflict resolution" below |
| `"small"` / `"large"`   | `"combined"`      | RESERVED — see "Conflict resolution" below |

**Conflict resolution.** A learned RM and self-judge log-likelihoods are
two competing notions of "non-deterministic reward". Combining them adds
a third reward source that the user is unlikely to want. For Phase 2
ship the simple rule:

> When `reward_model_capacity != "none"` AND `reward_mode != "deterministic"`,
> raise `ValueError` from `make_reward_fn` (or the new `build_reward_model`
> facade) with the message `"reward_mode='{mode}' incompatible with
> reward_model_capacity='{cap}'; choose one continuous reward source."`

The `reward_blend_alpha` knob already covers the "blend a learned RM with
the deterministic verifier" use case, so banning the multi-RM combination
costs no expressiveness. Lift the restriction in a future spec only if a
concrete experiment demands it.

### Adapter contract

`PPOTrainer.train_step` and `evaluate` invoke `self.reward_fn(completion,
ground_truth)` per sample today. The new `RewardModelScorer.score_batch`
takes lists. To keep the trainer's call sites unchanged when
`reward_model_capacity == "none"`, the cleanest design is:

1. `load_ppo_trainer` builds `reward_fn` exactly as today via
   `make_reward_fn(...)` ([ppo_trainer.py:1173-1175](../ppo_trainer.py#L1173-L1175)).
2. When `reward_model_capacity != "none"`, `load_ppo_trainer` calls
   `build_reward_model(config, ..., base_model=reference_model)` and wraps
   the resulting `RewardModelScorer` in a thin `_ScoreBatchAsCallable`
   adapter so it satisfies the existing per-sample `(completion, gt) -> float`
   protocol. The adapter buffers up the batch inside `generate_rollouts`
   (one `score_batch` call per training step) and then dispenses scores
   index-by-index, mirroring the `_RewardFnWrapper._idx` convention.
3. Alternatively (preferred for performance): `generate_rollouts` and
   `evaluate` learn to call `self.reward_model_scorer.score_batch(...)`
   directly when present and fall back to the per-sample loop otherwise.
   This requires touching the two call sites at [ppo_trainer.py:343](../ppo_trainer.py#L343)
   and [ppo_trainer.py:1024](../ppo_trainer.py#L1024) and is the path
   Agent B should take.

Either way, the existing `reward_fn` slot stays. Tests that monkey-patch
`trainer.reward_fn` (e.g., the diagnostic-fn dance at
[run_e2_7.py:248-253](../run_e2_7.py#L248-L253)) keep working under
`reward_model_capacity == "none"` and are explicitly broken-on-purpose
under non-`none` capacity (the diagnostic test path always uses
`gsm8k_reward`, which the new design preserves; only the
training-reward path swaps to `score_batch`).

### Checkpoint hash invariant

[`checkpoint.py:_config_hash`](../checkpoint.py#L210-L230) MUST include
all five new RM fields, otherwise resuming a run that was trained with
a learned RM into a config with `reward_model_capacity="none"` will
silently accept the mismatch and produce gibberish.

**Current state (2026-05-04):** the `key_fields` dict at
[checkpoint.py:223-225](../checkpoint.py#L223-L225) already includes
`reward_model_capacity`, `reward_blend_alpha`, and
`reward_score_activation` (via `getattr` defaults so old checkpoints still
load). The two MISSING fields are:

```python
"reward_model_name": getattr(config, "reward_model_name", None),
"reward_model_reuse_reference": getattr(config, "reward_model_reuse_reference", False),
```

Agent A adds these two and the corresponding `PPOConfig` fields. This is
part of Agent A's deliverable, not a follow-up.

### Accuracy metric: gsm8k_reward stays the source of truth

When `reward_model_capacity != "none"`, the RM produces continuous scores
that are NOT a 0/1 accuracy. The reported `accuracy` metric must
continue to use `gsm8k_reward` regardless of the reward used for
training. This means there are TWO reward signals to log per step:

| Metric | Source | Used for |
|--------|--------|----------|
| `mean_reward` | `reward_fn` (RM, blended, or gsm8k) | Training; logged as `mean_reward` |
| `accuracy` | `gsm8k_reward` always | Reporting; logged as `accuracy` |

Sites to update (all logged via `compute_accuracy(...)` from
[eval/metrics.py](../../eval/metrics.py)):

| File | Line | Change |
|------|------|--------|
| [ppo_trainer.py:`train_step`](../ppo_trainer.py) | accuracy aggregation | Replace `[r.reward for r in batch.rollouts]` with a fresh `[gsm8k_reward(r.completion, gt) for r, gt in ...]` when `capacity != "none"` |
| [ppo_trainer.py:`evaluate`](../ppo_trainer.py) | rewards list | Always score with `gsm8k_reward` for the returned accuracy; the RM-driven reward only affects training, not reporting |
| [run_e2_7.py:print line](../run_e2_7.py) | step log | Print both `acc` (gsm8k) and `mean_reward` (training signal) |
| [run_e2_8.py:print line](../run_e2_8.py) | step log | Same |

Without this rule, switching to a learned RM makes the accuracy curve
unreadable — it conflates training signal with held-out correctness.

### File-level change list

| File | Change |
|------|--------|
| **NEW** `ppo_specs/reward_model.py` | `RewardModelScorer` class, `build_reward_model(capacity, config, device)` factory. Implements `none`/`small`/`large` tiers. Applies `reward_score_activation` squash inside `score_batch`. |
| [ppo_specs/config.py:57-60](../config.py#L57-L60) | Add the 6 config fields below the existing reward block (capacity, name, dtype, reuse_reference, blend_alpha, score_activation). |
| [ppo_specs/checkpoint.py:210-230](../checkpoint.py#L210-L230) | `_config_hash` already partially updated (3 of 5 fields). Add the 2 missing fields: `reward_model_name`, `reward_model_reuse_reference`. |
| [ppo_specs/ppo_trainer.py:154-244](../ppo_trainer.py#L154-L244) | `PPOTrainer.__init__`: accept `reward_model_scorer` kwarg. Cache `self.reward_model_scorer`. Per the Baseline invariant, when `reward_model_capacity == "none"` the trainer keeps using `self.reward_fn` exactly as today. |
| [ppo_specs/ppo_trainer.py:267-389](../ppo_trainer.py#L267-L389) | `generate_rollouts`: when an RM scorer is active, replace the per-sample `self.reward_fn(completion, gt)` call at [L343](../ppo_trainer.py#L343) with one batched `self.reward_model_scorer.score_batch(prompts, completions, gts)` call that returns `[B]`. The diagnostic accuracy column comes from `gsm8k_reward` regardless. |
| [ppo_specs/ppo_trainer.py:954-1035](../ppo_trainer.py#L954-L1035) | `evaluate`: same batched substitution at [L1024](../ppo_trainer.py#L1024). For the returned accuracy, always use `gsm8k_reward`. |
| [ppo_specs/ppo_trainer.py:1040-1197](../ppo_trainer.py#L1040-L1197) | `load_ppo_trainer`: load RM via `build_reward_model(...)`; if `reward_model_reuse_reference=True` and shapes match, pass `reference_model` in as the base so weights are shared. Reuse the `accelerator.main_process_first()` cache-warming pattern at [L1079-L1099](../ppo_trainer.py#L1079-L1099). |
| [ppo_specs/run_e2_7.py](../run_e2_7.py), [ppo_specs/run_e2_8.py](../run_e2_8.py) | No structural change. Log `reward_model_capacity` in run header. |
| **NEW** `ppo_specs/tests/test_reward_model.py` | Unit tests (mock RM, shape/dtype/device asserts, `none` tier parity with `gsm8k_reward`). |
| **NEW** `ppo_specs/run_e2_rm.py` (optional) | E2.8-style sweep across `reward_model_capacity ∈ {none, small}` for smoke; `{none, small, large}` on the GPU cluster. |
| [ppo_specs/specs/memory_optimization.md](memory_optimization.md) | Append a section on 3-model coexistence (§6 below). |

## Memory profile (the 8B × 3 problem)

Three 8B models on one GPU:

| Component | bf16 weights | bf16 grads | AdamW states (fp32, 2×) | Activations (approx.) |
|-----------|--------------|------------|-------------------------|-----------------------|
| Policy (trainable) | 16 GB | 16 GB | 64 GB | ~10 GB |
| Reference (frozen) | 16 GB | 0 | 0 | ~1 GB (no-grad fwd) |
| Reward model (frozen) | 16 GB | 0 | 0 | ~1 GB (no-grad fwd) |
| **Total** | **48 GB** | 16 GB | 64 GB | **~12 GB** |

Rough total for training: **~140 GB**. Does not fit on a single 80 GB A100 or H100 without intervention.

### Mitigations (apply in this order, stop when it fits)

1. **`gradient_checkpointing=True` on policy**: drops policy activations from ~10 GB to ~4 GB. Already supported via [ppo_specs/config.py:106](../config.py#L106) and wired in [ppo_specs/ppo_trainer.py:1103-1107](../ppo_trainer.py#L1103-L1107). First lever.
2. **`reward_model_reuse_reference=True`**: load the RM base weights **once**, use them for both the KL anchor and RM scoring. Saves 16 GB weights. Only valid when RM architecture ≡ reference architecture (same `model_name`). Implementation: factory accepts an optional `base_model` handle; RM becomes `base_model + value_head`. Reference reads log-probs from `base_model` directly. 
3. **Offload frozen models to CPU when idle**: the RM and reference are only used during `generate_rollouts` and at the start of `ppo_update` (for `kl_ref`). Move to CPU during the `K` gradient epochs via `accelerator.cpu` or manual `.to("cpu")` / `.to(device)` around the compute windows. Saves another 16 GB during the gradient loop. Call-site: around the K-epoch loop body in `train_step` ([ppo_specs/ppo_trainer.py:912-926](../ppo_trainer.py#L912-L926)).

   Transfer cost: 16 GB / 32 GB/s (PCIe 4.0) = 0.5 s per direction per
   model. Per PPO step: 2 directions × 2 models = 2 s overhead.
   At ~12 s/step (4× A100 DDP at 8B), that's 17% overhead. Worth it
   when it's the difference between OOM and fit. If using gradient
   accumulation across many micro-batches per PPO step, batch the
   off-on transitions to amortize.

4. **8-bit AdamW (bnb.optim.AdamW8bit)**: drops optimizer state from
   96 GB to 48 GB at 8B. Quality loss <0.5% per the bnb paper.
   See [memory_optimization.md §11.1](memory_optimization.md#111-8-bit-adamw-bitsandbytes).
   This is the single largest win and orthogonal to all other mitigations.
5. **Quantize the frozen reference and RM**: `BitsAndBytesConfig(load_in_8bit=True)`
   on each frozen model saves 8 GB per quantized model (16 → 8 GB).
   With NF4 quantization saves 12 GB per model (16 → 4 GB).
   Both reference and RM are no-grad → quantization restrictions on
   training do NOT apply. See
   [memory_optimization.md §11.2](memory_optimization.md#112-quantize-the-frozen-reference-model).
6. **Reduce batch size for 8B runs**: `batch_size` from 16 → 8 (global). Already doable via config; note the Accelerate divisibility constraint (`batch_size % num_processes == 0`).
7. **DeepSpeed ZeRO-2** (future): shards optimizer states + grads, saves ~64 GB. Out of scope for this spec but called out in [distributed.md](distributed.md).

### Updated mitigation savings table at 8B

| Stack | Peak memory | Fits 80 GB A100? |
|-------|------------:|:-----------------|
| Default (3 × 8B bf16, fp32 AdamW, no GC) | 167 GB | NO |
| + Gradient checkpointing | 89 GB | Yes (tight) |
| + 8-bit AdamW | 41 GB | Yes |
| + Reference int8 | 33 GB | Yes |
| + RM int8 | 25 GB | Yes (comfortable) |
| + Reuse reference (RM base ≡ reference) | 17 GB | Yes (very comfortable) |
| + CPU offload during K-epoch loop | ~12 GB | Fits A100 40GB / RTX 4090 |

After (1) + (2) + (3): **~60 GB peak** on an 80 GB A100. With (4) + (5)
added: **~25 GB peak** — fits a 40 GB A100. Full stack: **~12 GB peak**
during gradient phase, fits any modern GPU.

### CPU smoke memory

At the `small` tier (Qwen-0.5B RM), three 0.5B bf16 models + optimizer = ~14 GB. Fine on any CPU node with ≥16 GB RAM.

## Testing strategy

### Unit tests (`ppo_specs/tests/test_reward_model.py`)

1. `test_none_tier_parity`: `build_reward_model(capacity="none")` returns a scorer whose `score_batch` output matches `gsm8k_reward` on a hand-rolled batch, float-exact.
2. `test_small_tier_shape_dtype`: mock the HF load with a tiny `AutoModel.from_pretrained("sshleifer/tiny-gpt2")` (or similar), confirm `score_batch` returns `torch.Tensor` of shape `[B]`, dtype `float32`, device matches.
3. `test_blend_alpha_interpolation`: for `alpha=0`, output equals `gsm8k_reward`; for `alpha=1`, output equals RM scores; for `alpha=0.5`, element-wise midpoint. Float tolerance 1e-6.
4. `test_reuse_reference_weight_sharing`: when `reward_model_reuse_reference=True` and base matches, memory footprint is one model not two. Verify via `id(rm.base_model) == id(reference_model)` or parameter-pointer equality.
5. `test_ground_truth_optional`: passing `ground_truths=None` to a trained RM scorer works; `none` tier with `ground_truths=None` raises a clear error.

Skip CUDA-specific tests on CPU runners (use the existing `torch.cuda.is_available()` guard pattern from [ppo_specs/run_e2_8.py:234](../run_e2_8.py#L234)).

### Integration smoke test (CPU, `small` tier)

Extends the existing local-test smoke (see `local_test_config` in [ppo_specs/config.py:142-157](../config.py#L142-L157)):

```python
local_test_rm_config():
    # inherits from local_test_config(), overrides:
    reward_model_capacity = "small"
    reward_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # or a real RM checkpoint
    reward_blend_alpha = 1.0
```

Run via:
```bash
PYTHONUTF8=1 accelerate launch \
    --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_7.py --local-test --local-test-rm --no-mc
```

Expected: 5 steps, all ranks converge on identical gathered rewards, `final_acc` prints on rank 0, no hang on RM forward, no OOM.

### GPU migration test (`large` tier)

After CPU smoke passes:
```bash
sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4,REWARD_MODEL_CAPACITY=large \
    scripts/slurm_e2_7.sh
```

First GPU submission should set `reward_model_reuse_reference=True` + `gradient_checkpointing=True`, batch_size=8. Monitor `nvidia-smi`/WandB memory metrics on the first step; abort and retry with smaller batch if >75 GB used.

## Risks & gotchas

- **RM inference under DDP**: use `accelerator.unwrap_model(rm)` before calling the RM's forward — same subtlety as policy `.generate()` (see distributed.md §3.2). Avoid DDP sync hangs on frozen models.
- **RM must be frozen**: `requires_grad_(False)` on every parameter at load time. Mirror the reference-model freeze pattern at [ppo_specs/ppo_trainer.py:194-201](../ppo_trainer.py#L194-L201) and the explicit `requires_grad_(False)` loop at [ppo_specs/ppo_trainer.py:1170-1171](../ppo_trainer.py#L1170-L1171).
- **Scalar output scale**: a raw value-head output can be unbounded; the existing PPO advantage normalization ([ppo_specs/advantage.py](../advantage.py)) assumes rewards in a reasonable range. Consider tanh or clip the RM output to `[0, 1]` or `[-1, 1]` to match GSM8K-reward scale, especially with `reward_blend_alpha ∈ (0, 1)` where both signals are added.
- **Tokenizer mismatch**: if the RM uses a different tokenizer than the policy, the `score_batch` implementation must re-tokenize from strings (not reuse policy token ids). This is the robust default — implement it that way.
- **`reward_nonzero_rate` diagnostic**: the existing logging ([ppo_specs/run_e2_7.py:303](../run_e2_7.py#L303)) assumes binary rewards. With a continuous RM, this metric becomes meaningless. Either rename to `reward_mean`/`reward_variance` in the RM path or keep it for the `none` tier only.
- **CPU bf16**: Accelerate CPU mode uses `mixed_precision: no` per the accelerate_cpu plan. RM load should respect `device.type` to pick fp32 on CPU even when `reward_model_dtype="auto"`.
- **Checkpoint hash**: [ppo_specs/checkpoint.py](../checkpoint.py) `_config_hash` must include the new reward-model fields so a resume across different RM configs fails loudly.

## Verification checklist

- [ ] Unit tests pass on CPU: `pytest ppo_specs/tests/test_reward_model.py -v`
- [ ] Full test suite passes: `pytest ppo_specs/tests/ -v`
- [ ] `none` tier CPU smoke matches pre-RM baseline: `PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc` — final_acc numerically identical to pre-change run on the same seed
- [ ] `small` tier CPU smoke completes without hang or OOM
- [ ] `small` tier multi-process CPU DDP smoke (4 procs) completes; gathered rewards are identical across ranks
- [ ] Memory audit on 8B path confirms <75 GB peak on A100-80GB with mitigations (1)+(2)+(3)
- [ ] `large` tier first GPU run completes 5 steps without OOM
- [ ] Checkpoint save/load round-trip preserves RM config (config hash matches)
