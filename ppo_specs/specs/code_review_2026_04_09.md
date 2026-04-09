# Code Review — 2026-04-09

Comprehensive review by three specialized agents: training logic (DL PhD),
readability (senior SWE), and performance (ML infrastructure).

---

## Executive Summary

| Agent | Critical | Important/Major | Moderate/Medium | Minor/Low |
|-------|----------|----------------|-----------------|-----------|
| Training Logic | 0 (12 verified correct) | 6 | — | 4 |
| Readability | — | 4 high | 9 medium | 7 low |
| Performance | 2 | 4 major | 5 moderate | 5 minor |

**Core PPO algorithm is correct.** All loss functions, gradient isolation, advantage
computation, ratio clipping, and KL direction verified. The main concerns are
scaling to 8B models (OOM without micro-batching/FSDP), redundant forward passes
(6 per step, reducible to 3), and several code quality issues now fixed.

---

## Fixes Applied in This Review

| File | Fix | Category | Ref |
|------|-----|----------|-----|
| `ppo_trainer.py` | Removed dead legacy methods `_sequence_log_prob`, `_critic_value_no_grad` | Readability H2 |
| `ppo_trainer.py` | Extracted `_extract_last_hidden()` to deduplicate 3-way critic hidden-state code | Readability H1 |
| `ppo_trainer.py` | `max_length=512` → `config.max_prompt_length` (5 sites) | Readability H4 |
| `ppo_trainer.py` | `max_norm=1.0` → `config.grad_clip_norm` (2 sites) | Readability M6 |
| `ppo_trainer.py` | `-20.0/20.0` log-ratio clamp → `config.log_ratio_clip` | Readability M5 |
| `ppo_trainer.py` | Eval batch size → `config.eval_batch_size` | Readability M7 |
| `ppo_trainer.py` | Removed unused `field` import | Readability L5 |
| `config.py` | Added `max_prompt_length`, `grad_clip_norm`, `log_ratio_clip`, `eval_batch_size` | Config |
| `config.py` | Added class-level docstring to PPOConfig | Readability R12 |
| `run_e2_7.py` | Fixed logger overwrite on resume (was unconditionally recreated) | Training MINOR-3 |
| `run_e2_7.py` | Renamed `reward_window` → `reward_history` | Readability R6 |
| `run_e2_8.py` | Reset RNG seeds before each capacity run for fair comparison | Training IMPORTANT-5 |
| `run_e2_8.py` | Replaced `ALL_CAPACITIES` with `CRITIC_CAPACITIES` import from config | Readability M3/R10 |
| `run_e2_8.py` | Fixed GPU memory cleanup: `model.cpu()` + `gc.collect()` before `empty_cache()` | Safety S8 |
| `test_batched_ops.py` | Fixed `single_tensor` → `individual_tensor` (2 sites, NameError) | Readability M4 |

---

## Verified Correct (Training Logic)

These items were rigorously verified by the DL PhD review agent:

| Item | Location | Why Correct |
|------|----------|-------------|
| PPO-clip surrogate `−E[min(ρA, clip(ρ)A)]` | `ppo_trainer.py:393-399` | Standard PPO-clip with correct negation for minimization |
| Sequence log-prob indexing `logits[pl-1:seq_len-1]` | `ppo_trainer.py:240-243` | Logits at t predict token t+1; verified boundary is exact |
| KL direction `(old_lp − new_lp)` = KL(π_old ‖ π_new) | `ppo_trainer.py:402-405` | Correct: penalizes new policy moving from old under which data was collected |
| Critic gradient isolation | `ppo_trainer.py:476-487` | `torch.no_grad()` + `.detach()` prevents critic loss backprop into policy |
| Policy gradient isolation from critic | `ppo_trainer.py:408-416` | Separate optimizer param sets; verified no cross-contamination |
| GAE = r − V(s) for single-step | `advantage.py:42-46` | Terminal reward → V(s_terminal) = 0 → GAE = r − V(s) |
| Advantages precomputed once before K epochs | `ppo_trainer.py:507-523` | Matches Schulman 2017 Section 4 |
| Advantage normalization edge cases | `advantage.py:48-56` | Handles zero-std (all-zero output) and single-element correctly |
| fp32 for log_softmax | `ppo_trainer.py:229` | Prevents catastrophic cancellation in bf16 |
| Log-ratio clamping | `ppo_trainer.py:394` | exp(20) ≈ 5e8; PPO clip at ε=0.2 handles the rest |
| Checkpoint atomicity | `checkpoint.py:28-97` | Write to tmp dir then rename; RNG states saved |
| Left-padding strip in generation | `ppo_trainer.py:166-170` | Correctly strips pad tokens from output before building Rollout |

---

## Open Issues — Training Logic

### TL-1: `n_ppo_epochs=1` underutilizes PPO (IMPORTANT)
**Location:** `config.py:27`
**Issue:** With K=1, PPO's key efficiency gain (reusing rollout data for multiple
gradient updates) is lost. The clipping mechanism provides minimal benefit when
the ratio starts at 1.0 and barely moves in one step.
**Recommendation:** Consider K=3-4 for E2.7/E2.8 configs. Standard PPO uses K=3-10.

### TL-2: MC baselines estimated once from initial policy (IMPORTANT)
**Location:** `run_e2_7.py:129-137`, `run_e2_8.py:209-216`
**Issue:** MC baselines are computed once at training start. As PPO changes the
policy, the advantage estimation error metric (iv) measures tracking of the
*initial* value function, not the current one. Documented in code comments and
logic.md L6.
**Recommendation:** Re-estimate MC baselines every 50 steps for longer runs.

### TL-3: Empty response log-prob of 0.0 semantically wrong (IMPORTANT)
**Location:** `ppo_trainer.py:238-239`
**Issue:** When response is empty, `log_prob = 0` implies `probability = 1.0` for
generating nothing. In practice this is rare and produces benign ratio=1.0 with
zero advantage contribution.
**Status:** Low probability of triggering. Documented in safety.md S10.

### TL-4: No data shuffling between epochs (MINOR)
**Location:** `utils.py:22-40`
**Issue:** `cycle_batch` repeats the same batch order every
`n_train_samples/batch_size` steps. For 200 samples and batch_size=16, this
repeats every ~12 steps.
**Recommendation:** Add epoch-level shuffling for runs longer than 200 steps.

### TL-5: Logger double-create on resume — FIXED
**Status:** Fixed in this review. The logger is now only created when not resuming.

### TL-6: E2.8 seed reset between capacities — FIXED
**Status:** Fixed in this review. Seeds are now reset before each capacity run.

---

## Open Issues — Performance

### PERF-1: 8B models will OOM without micro-batching (CRITICAL)
**Impact:** Cannot train at all on a single A100 80GB.
**Analysis:** Optimizer states alone for 8B full fine-tuning = 96 GB (AdamW:
master weights + momentum + variance, all fp32). With weights (16 GB bf16) +
gradients (16 GB) + activations = ~134 GB.
**Fix required:** Either FSDP to shard optimizer across 2+ GPUs, or LoRA
(reduces to ~17 GB, fits on A100 40GB).

### PERF-2: No multi-GPU / distributed support (CRITICAL)
**Status:** Open (P7 in performance.md)
**Fix required:** Integrate HuggingFace `accelerate` with FSDP ZeRO-3.

### PERF-3: Generation logits discarded then recomputed (MAJOR)
**Location:** `ppo_trainer.py:154-161` → `ppo_trainer.py:188-193`
**Impact:** 1 wasted batched full-sequence forward pass per step.
**Fix:** Use `model.generate(..., output_scores=True, return_dict_in_generate=True)`.

### PERF-4: Separate critic+policy forward passes in PPO update (MAJOR)
**Location:** `ppo_trainer.py:438-448` and `ppo_trainer.py:450-491`
**Impact:** K extra prompt-only forward passes per step.
**Fix:** Single full-sequence forward with `output_hidden_states=True` providing both.

### PERF-5: Redundant critic forward for advantage precomputation (MAJOR)
**Location:** `ppo_trainer.py:510-513`
**Impact:** 1 extra prompt-only forward pass per step.
**Fix:** Use critic values from first PPO epoch's `_critic_forward` call.

### PERF-6: Full-vocabulary log_softmax tensor ~7 GB (MAJOR)
**Location:** `ppo_trainer.py:229-230`
**Impact:** `[B, max_len, V]` tensor in fp32. For B=16, S=768, V=152K: 7.1 GB.
**Fix:** Fuse `log_softmax` + `gather` to avoid materializing full tensor.

### PERF-7: Model re-loaded from disk per capacity in E2.8 (MODERATE)
**Fix:** Load once, deep-copy `state_dict()`, `load_state_dict()` per run.

### Forward pass count per `train_step` (current)
```
4 + 2K batched forward passes (K = n_ppo_epochs)
K=1: 6 passes, K=4: 12 passes

After fixing PERF-3/4/5: 2 + K passes
K=1: 3 passes (2x reduction), K=4: 6 passes (2x reduction)
```

---

## Open Issues — Readability

### READ-1: MC baseline code duplicated in run scripts (HIGH)
**Location:** `run_e2_7.py:124-137`, `run_e2_8.py:205-216`
**Fix:** Replace with `setup_mc_baselines()` from `utils.py`.

### READ-2: `_tiny_config()` duplicated in 5 test files (MEDIUM)
**Fix:** Define once in `tests/conftest.py`.

### READ-3: Inconsistent type annotations (MEDIUM)
**Fix:** Use modern `list[T]`, `dict[K,V]` throughout (Python 3.11 confirmed).

### READ-4: Variable name `B` for batch size (LOW)
**Location:** `ppo_trainer.py:144`
**Fix:** Rename to `n_prompts` or `batch_size`.

---

## Memory Estimates

### Qwen2.5-0.5B (development)
| Component | Size |
|-----------|------|
| Weights (bf16) | 1 GB |
| Optimizer (AdamW fp32) | 6 GB |
| Gradients (bf16) | 1 GB |
| Activations (B=16, S=768) | ~6.4 GB |
| log_softmax tensor | ~7.1 GB |
| **Peak total** | **~21.5 GB** |
| With gradient checkpointing | ~14 GB |

### Llama-3-8B (cluster)
| Component | Size |
|-----------|------|
| Weights (bf16) | 16 GB |
| Optimizer (AdamW fp32) | 96 GB |
| Gradients (bf16) | 16 GB |
| Activations (B=16, S=768, GC) | ~6.3 GB |
| log_softmax tensor | ~6.0 GB |
| **Peak total** | **~140 GB** (does NOT fit 1x A100 80GB) |
| With FSDP across 2 GPUs | ~70 GB/GPU |
| With LoRA | ~17 GB total |
