# Scaling and Training Logic Analysis

Critical analysis of the PPO implementation's ability to scale to large compute
systems and correctness of the training pipeline at scale.

---

## 1. Training Logic Correctness — Summary

### 1.1 What Is Verified Correct

The core PPO algorithm is mathematically sound. Every critical component has been
traced through the gradient computation graph:

**Loss function chain:**
```
total_loss = policy_loss + critic_loss_coeff * critic_loss + kl_coeff * kl

policy_loss → model.parameters() ✓  (through _policy_log_probs → model forward with grad)
critic_loss → critic.parameters() ✓  (model forward under no_grad, hidden states detached)
kl          → model.parameters() ✓  (through new_log_probs → model forward with grad)

Cross-term: critic_loss ╳→ model.parameters()  (verified: no_grad + detach blocks flow)
Cross-term: policy_loss ╳→ critic.parameters()  (verified: critic not in policy graph)
```

**PPO-clip surrogate (verified):**
```python
ratio = exp(log_π_θ(a|s) - log_π_θ_old(a|s))          # correctly uses stored old log-probs
clipped = clamp(ratio, 1-ε, 1+ε)                        # standard PPO clip
L_CLIP = -E[min(ratio * A, clipped * A)]                # negated for PyTorch minimization
```

**GAE for single-step RLVR (verified):**
```
Episode: s=prompt → a=response → r=binary{0,1} → terminal
V(s_terminal) = 0 (episode ends)
GAE = δ_0 = r + γ·V(s_1) - V(s_0) = r - V(s)  ✓
```

**Advantage handling (verified):**
- Computed ONCE before K PPO epochs (Schulman 2017 §4)
- Normalized with z-score, handles zero-std edge case
- When critic="none", batch-mean baseline used instead

### 1.2 Known Limitations Affecting Training Quality

| Issue | Impact | Severity |
|-------|--------|----------|
| `n_ppo_epochs=1` | PPO's multi-epoch reuse disabled; clipping mechanism underutilized | Medium |
| MC baselines from initial policy only | Advantage error metric becomes stale after ~50 steps | Low (metric only) |
| No data shuffling | Same batch order repeats every ~12 steps (200 samples, B=16) | Low |
| No learning rate warmup | First steps may have large gradient updates | Low (lr=1e-5 is conservative) |
| No entropy regularization | Exploration limited to temperature sampling only | Low (intentional for RLVR) |

### 1.3 Numerical Stability at Scale

| Concern | Current Protection | Adequate? |
|---------|-------------------|-----------|
| bf16 log_softmax | Logits upcast to fp32 before log_softmax | Yes |
| Log-ratio overflow | Clamped to ±20 before exp() | Yes (configurable) |
| Zero-std advantages | All-zero output when std < 1e-8 | Yes |
| Long response log-probs | Sum of ~200-400 token log-probs (range -200 to -600) | Yes (difference cancels) |
| Gradient explosion | clip_grad_norm at 1.0 (now configurable) | Yes |

---

## 2. Scaling Analysis

### 2.1 Forward Pass Budget

Current cost per `train_step` with K = n_ppo_epochs:

```
Phase 1 (rollout generation, no grad):
  1  batched model.generate()         — autoregressive, ~T steps
  1  batched full-sequence forward     — log-prob computation
  1  batched prompt-only forward       — critic values (skipped for "none")

Phase 1.5 (advantage precomputation, no grad):
  1  batched prompt-only forward       — critic values (REDUNDANT with Phase 2)

Phase 2 (PPO update, per epoch × K):
  1  batched prompt-only forward       — critic hidden states (no_grad on model)
  1  batched full-sequence forward     — policy log-probs (WITH grad)

Total: 4 + 2K  batched forward passes
  K=1: 6 passes    K=4: 12 passes
```

**Optimization roadmap (see PERF-3/4/5 in code_review_2026_04_09.md):**
```
After reusing generation logits:     3 + 2K  (saves 1 full-seq pass)
After merging critic+policy forward: 3 + K   (saves K prompt-only passes)
After removing redundant precompute: 2 + K   (saves 1 prompt-only pass)

Optimized total: 2 + K
  K=1: 3 passes (2x reduction)
  K=4: 6 passes (2x reduction)
```

### 2.2 Memory Scaling

#### Qwen2.5-0.5B (development target)

| Component | bf16 | fp32 |
|-----------|------|------|
| Model weights | 1 GB | 2 GB |
| Optimizer states (AdamW) | 6 GB | 6 GB |
| Gradients | 1 GB | 2 GB |
| Activations (B=16, S=768) | ~3 GB | ~6.4 GB |
| log_softmax peak | ~7.1 GB (always fp32) | ~7.1 GB |
| **Peak GPU** | **~18 GB** | **~24 GB** |
| With gradient checkpointing | ~11 GB | ~14 GB |

**Verdict:** Fits on RTX 3090/4090 (24 GB) with bf16. Comfortable with GC.

#### Llama-3-8B (cluster target)

| Component | bf16 | fp32 |
|-----------|------|------|
| Model weights | 16 GB | 32 GB |
| Optimizer states (AdamW, fp32) | 96 GB | 96 GB |
| Gradients | 16 GB | 32 GB |
| Activations (B=16, S=768, GC) | ~6.3 GB | ~12.6 GB |
| log_softmax peak | ~6.0 GB | ~6.0 GB |
| **Peak GPU (1 GPU)** | **~140 GB** | **~179 GB** |

**Verdict:** Does NOT fit on a single A100 80GB. Requires one of:

1. **FSDP across 2× A100 80GB** — shards optimizer states + weights across GPUs.
   ~70 GB/GPU. Manageable.

2. **LoRA (r=16)** — Only 0.1% of parameters trainable. Optimizer states drop
   from 96 GB to ~0.1 GB. Total: ~17 GB. Fits on A100 40GB.

3. **Gradient accumulation** — Reduces batch from B=16 to micro_batch=2.
   Activations drop but optimizer states unchanged. Still OOM on 1 GPU.

**Recommended path for cluster:** LoRA for single-GPU, FSDP for multi-GPU.

### 2.3 Throughput Estimates

Estimated wall-clock per `train_step` (B=16 prompts, T=384 max tokens):

| Model | Device | Generate | Log-probs | Critic | PPO Update | Total |
|-------|--------|----------|-----------|--------|------------|-------|
| Qwen 0.5B | RTX 4090 | ~2s | ~0.1s | ~0.1s | ~0.3s | ~2.5s |
| Qwen 0.5B | A100 | ~1s | ~0.05s | ~0.05s | ~0.2s | ~1.3s |
| Llama 8B | A100 80GB (FSDP 2x) | ~8s | ~1s | ~0.3s | ~2s | ~11s |
| Llama 8B | A100 80GB (LoRA) | ~5s | ~0.5s | ~0.2s | ~0.5s | ~6s |

**Total training time (200 steps + eval):**

| Model | Device | Training | Eval | MC baselines | Total |
|-------|--------|----------|------|--------------|-------|
| Qwen 0.5B | A100 | ~4 min | ~2 min | ~3 min | ~9 min |
| Llama 8B | 2× A100 FSDP | ~37 min | ~10 min | ~20 min | ~67 min |
| Llama 8B | 1× A100 LoRA | ~20 min | ~5 min | ~10 min | ~35 min |

### 2.4 Multi-GPU Considerations

The codebase currently has no distributed training support. To scale:

**Required changes for FSDP:**
1. Wrap model/optimizer with `accelerate.prepare()`
2. Gate checkpoint saves on `accelerator.is_main_process`
3. Replace explicit `.to(device)` with accelerate device placement
4. Add `dist.barrier()` around checkpoint save/load
5. Aggregate metrics with `accelerate.gather()` before logging

**Global state that needs coordination:**
- `PPOTrainer.step`, `PPOTrainer.total_rollouts` — must be consistent across ranks
- `ExperimentLogger` — only rank 0 should write
- `GracefulExitHandler` — signal handling on rank 0, broadcast to others
- `cycle_batch` — deterministic across ranks (same seed guarantees this)

**Checkpoint changes:**
- Only rank 0 saves model/optimizer via `accelerator.save_state()`
- All ranks must call `dist.barrier()` after save before rotation
- Resume must load on all ranks simultaneously

---

## 3. Critical Path Items for Cluster Deployment

### Priority 1: Must Fix (blocking)
| Item | Effort | Impact |
|------|--------|--------|
| Add LoRA support OR FSDP | 2-3 hours | Enables 8B training |
| Gradient accumulation in `_policy_log_probs` | 1 hour | Reduces per-GPU memory |

### Priority 2: Should Fix (significant improvement)
| Item | Effort | Impact |
|------|--------|--------|
| Reuse generation logits (PERF-3) | 2 hours | Saves 1 full-seq fwd pass/step |
| Merge critic+policy forward (PERF-4/5) | 3 hours | Saves K+1 fwd passes/step |
| Fuse log_softmax+gather (PERF-6) | 1 hour | Saves ~7 GB peak memory |
| Increase `n_ppo_epochs` to 4 | 5 min | Better PPO utilization |

### Priority 3: Nice to Have
| Item | Effort | Impact |
|------|--------|--------|
| Re-estimate MC baselines periodically | 1 hour | More accurate metrics |
| Data shuffling per epoch | 30 min | Better generalization |
| Learning rate warmup | 30 min | Safer first few steps |
| Model state_dict caching in E2.8 | 1 hour | Faster sweep setup |

---

## 4. Experimental Methodology Assessment

### E2.7: Head-to-Head Comparison
**Strengths:**
- All four measurements (i-iv) properly implemented and logged
- Matched compute via `n_rollouts_per_prompt=1` for PPO
- Multiple seeds supported via CLI
- MC baselines provide ground-truth reference

**Weaknesses:**
- MC baselines go stale (initial policy only) — documented in code
- `n_ppo_epochs=1` means PPO is not using its full advantage
- Eval noise with small `eval_size` (configurable, default now 100)

### E2.8: Critic Quality Sweep
**Strengths:**
- All four capacities properly implemented with distinct architectures
- εV (RMSE) is the correct metric for crossover analysis
- MC baselines shared across capacities for fair comparison
- Each capacity starts from fresh pretrained weights

**Weaknesses (now fixed):**
- ~~Seeds were not reset between capacity runs~~ FIXED
- ~~GPU memory not freed between runs~~ FIXED

**Remaining weakness:**
- The crossover prediction (Theorem 2.5) assumes known σ²_A which isn't
  measured directly. The empirical crossover from εV vs accuracy curves
  should be used instead.

---

## 5. Recommendations for Publication

1. **Run with `n_ppo_epochs=4`** to properly utilize PPO's multi-epoch update.
   This is the standard configuration and may significantly change results.

2. **Report memory requirements** for each model size. The current implementation
   cannot train 8B without LoRA or FSDP.

3. **Re-estimate MC baselines** at least at steps {0, 50, 100, 150, 200} to
   demonstrate that advantage estimation error is measured against the current
   policy, not just the initial one.

4. **Use 3+ seeds** for each experiment cell. The E2.8 seed reset fix ensures
   fair comparison across capacities within a seed.

5. **Report clip fraction** as a diagnostic. High clip fraction (>0.3) suggests
   the policy is changing too fast; low (<0.01) suggests PPO clip is not active
   (equivalent to REINFORCE with K=1).
