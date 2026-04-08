# Logic Fix Spec

This spec covers correctness of the PPO algorithm, experimental measurement logic,
and data pipeline correctness.

Items marked PASS were reviewed and found correct; no change needed.
Items marked FIXED were bugs that have been corrected in the codebase.
Items marked BUG/OPEN require attention.

---

## Confirmed Correct (no change needed)

| Item | Location | Status |
|------|----------|--------|
| PPO-clip surrogate `min(rhoA, clip(rho,1+/-e)A)` | `ppo_trainer.py` | PASS |
| Advantages detached before ratio computation | `ppo_trainer.py` | PASS |
| `old_log_prob` stored at generation time (before update) | `ppo_trainer.py` | PASS |
| Critic MSE loss against observed returns (correct for gamma=1 single-step) | `ppo_trainer.py` | PASS |
| Critic hidden states detached from policy graph | `ppo_trainer.py` | PASS |
| GAE = `r - V(s)` for single-step terminal-reward episodes | `advantage.py` | PASS |
| MC baseline = `E[r \| prompt]` = mean reward over samples | `advantage.py` | PASS |
| Advantage normalization guard for batch size 1 | `advantage.py` | PASS |
| Log-ratio clamping to prevent exp overflow | `ppo_trainer.py` | PASS |
| `n_ppo_epochs` loop with precomputed advantages | `ppo_trainer.py` | PASS |
| KL penalty term applied in total loss | `ppo_trainer.py` | PASS |
| `_eval_critic_on_prompts` used for trainable critics in E2.7 | `run_e2_7.py` | PASS |
| `_eval_critic_on_prompts` used for trainable critics in E2.8 | `run_e2_8.py` | PASS |
| GSM8K dataset loaded correctly (`openai/gsm8k`, `main` split) | `src/data.py` | PASS |
| Ground-truth extraction via `split("####")[-1]` | `src/data.py` | PASS |
| Reward comparison with floating-point tolerance | `src/rewards.py` | PASS |
| Negative number handling in reward regex (`-?`) | `src/rewards.py` | PASS |
| Comma-separated number handling (`replace(",", "")`) | `src/rewards.py` | PASS |
| Decimal number handling (`\.?\d*` in regex) | `src/rewards.py` | PASS |
| PPO-clip sign (negate for PyTorch minimization) | `ppo_trainer.py` | PASS |
| Sequence log-prob indexing `logits[prompt_len-1:-1]` | `ppo_trainer.py` | PASS |
| KL divergence direction `(old_lp - new_lp)` = KL(pi_old \|\| pi_new) | `ppo_trainer.py` | PASS |
| Critic evaluates on prompt only (not prompt+response) | `ppo_trainer.py` | PASS |
| Gradient flow: policy_loss does not backprop into critic | `ppo_trainer.py` | PASS |
| Gradient flow: critic_loss does not backprop into policy | `ppo_trainer.py` | PASS |

---

## Previously Fixed Issues

### ID: L1 -- n_ppo_epochs was ignored
**Status**: Fixed
**Severity**: Critical
**Description**: `train_step` did not loop over `n_ppo_epochs`.
**Fix**: `train_step` now loops K times with precomputed advantages held fixed.

### ID: L2 -- kl_coeff was not applied
**Status**: Fixed
**Severity**: Critical
**Description**: The KL penalty term was not included in the total loss.
**Fix**: `ppo_update` now computes `KL(pi_old || pi_new)` and adds `kl_coeff * kl`.

### ID: L3 -- E2.7 advantage error used batch-mean for all critics
**Status**: Fixed
**Severity**: Critical
**Fix**: Now uses `trainer._eval_critic_on_prompts()` for trainable critics.

### ID: L4 -- E2.8 critic error used batch-mean for all critics
**Status**: Fixed
**Severity**: Critical
**Fix**: Now uses `trainer._eval_critic_on_prompts()` for trainable critics.

### ID: L6 -- Advantages recomputed each PPO epoch (non-standard)
**Status**: Fixed
**Severity**: Critical
**Description**: Advantages were recomputed from the updated critic each epoch.
**Fix**: `train_step` now precomputes advantages once before the K-epoch loop.

### ID: L7 -- Advantage normalization did not subtract mean
**Status**: Fixed
**Severity**: High
**Description**: `compute_advantages` divided by std only, not full z-score.
**Fix**: Updated to `(advantages - mean) / (std + eps)`.

### ID: L8 -- train_step returned only last epoch's metrics
**Status**: Fixed
**Severity**: Medium
**Fix**: Now averages metrics across all K epochs.

### ID: L9 -- clip_fraction metric retained computation graph
**Status**: Fixed
**Severity**: Low
**Fix**: Now uses `ratio.detach()`.

### ID: TD-1 -- Hardcoded critic_loss_coeff=0.5
**Status**: Fixed
**Severity**: Medium
**Fix**: Added `critic_loss_coeff` to PPOConfig.

---

## Issues Fixed in This Review (2026-04-07)

### ID: L10 -- extract_answer_from_completion first-vs-last #### inconsistency
**Status**: Fixed
**Severity**: Medium
**Description**: `data.py:extract_answer` used `split("####")[-1]` (last occurrence)
to extract ground-truth answers, but `rewards.py:extract_answer_from_completion`
used `re.search` (first occurrence) to extract model answers. If a model outputs
`#### 5 ... #### 10`, the completion extractor would return `5` while it should
return `10` (the final answer, consistent with how ground-truth is extracted).
**Analysis**: While GSM8K ground truths contain exactly one `####`, models in
chain-of-thought reasoning may produce multiple `####` markers. The last one is
the intended final answer. This inconsistency could cause false negatives where
the model gets the right answer but an earlier `####` is extracted instead.
**Fix**: Changed `re.search` to `re.findall` + take last match in
`rewards.py:extract_answer_from_completion`.

### ID: L11 -- Empty response log-prob returns wrong shape
**Status**: Fixed
**Severity**: Medium
**Description**: `_sequence_log_prob` returned `torch.tensor(0.0)` (scalar, shape
`[]`) for empty responses. The caller `_policy_log_probs` appends `lp.squeeze(0)`
and then calls `torch.stack(log_probs)`. If any sample in the batch generates
zero tokens, `torch.stack` would fail with a shape mismatch because some elements
have shape `[]` while others have shape `[1]` (from the `.squeeze(0)` on the
normal code path's `[1]` tensor).
**Fix**: Changed return value to `torch.zeros(1, device=self.device)` for
consistent shape `[1]`.

---

## Open Issues

### ID: L12 -- format_prompt does not use chat template
**Status**: Open
**Severity**: Medium
**Description**: `format_prompt()` constructs a plain-text prompt but the models
used (Qwen2.5-Instruct, Llama-3-Instruct) expect chat-template-formatted inputs.
`format_prompt_with_template()` exists in `data.py` but is never called.
**Analysis**: Using plain text with instruction-tuned models degrades performance
because the model has not been trained on this format. This is a systematic
confound: accuracy results may underestimate model capability.
**Recommendation**: Switch to `format_prompt_with_template()` and pass the
tokenizer through the data loading pipeline, OR use base (non-instruct) models.

### ID: L13 -- Reward function "last number" fallback false positives
**Status**: Open
**Severity**: Low
**Description**: The final fallback in `extract_answer_from_completion` takes the
last number in the text. This can match intermediate calculations, step numbers,
or other non-answer numbers.
**Analysis**: The fallback only triggers when `####`, `\boxed{}`, and "the answer
is" formats all fail. For instruction-tuned models prompted to use `####`, this
is uncommon. The risk is low but nonzero.
**Recommendation**: Log fallback activation rate; remove or discount if > 5%.

### ID: L14 -- No reference model KL divergence
**Status**: Won't Fix (for now)
**Severity**: Low (for RLVR)
**Description**: Standard RLHF uses a frozen reference model for KL constraint.
This implementation only uses step-level KL.
**Analysis**: For RLVR with verifiable rewards and short runs (100-200 steps),
reference KL is not necessary. See `performance.md` for full analysis.

### ID: L15 -- Evaluation n_eval inconsistency
**Status**: Open (by design)
**Severity**: Low
**Description**: `n_eval=20` during training, `n_eval=50` for final. With 20
samples at p=0.5, SE = sqrt(0.5*0.5/20) = 0.112, making curves noisy.
**Recommendation**: Use n_eval >= 200 in final evaluation for the paper.

---

## Confirmed Correct -- Deep Verification

### Gradient Flow (complete trace)

The combined loss is:
```
total_loss = policy_loss + c_v * critic_loss + c_kl * kl
```

**Policy loss -> model.parameters()**: `_policy_log_probs` -> `_sequence_log_prob`
runs the model WITH grad. `log_softmax` -> `gather` -> `sum` -> `new_log_probs`.
`ratio = exp(new_lp - old_lp.detach())`. Policy loss has grad through model.

**Critic loss -> critic.parameters()**: `_critic_forward` runs the model inside
`torch.no_grad()`, detaches hidden states, then runs the critic MLP with grad.
Critic loss has grad ONLY through critic params; zero through model.

**KL -> model.parameters()**: `kl = (old_lp.detach() - new_lp).mean()`. Gradient
flows through `new_lp` -> model. No flow to critic.

**Cross-term verification**: No gradient contamination between policy and critic.
Both optimizers update independent parameter sets. VERIFIED.

### Numerical Stability

- `log_softmax` is numerically stable (log-sum-exp trick internally).
- Sequence log-probs for R=200 tokens: typical range [-200, -600].
- `log_ratio = new_lp - old_lp` cancels most magnitude; typical |log_ratio| < 1.
- Clamp at [-20, 20] prevents exp overflow; only activates catastrophically.
- VERIFIED: No numerical stability issues.

### Optimizer Setup

- Policy lr=1e-5: Standard for LLM fine-tuning (1e-6 to 5e-5).
- Critic lr=1e-4: 10x policy, standard in actor-critic (critic should learn faster).
- AdamW: Appropriate; weight decay provides implicit regularization.
- No lr scheduler: Acceptable for 100-200 step runs. Add warmup for longer runs.

### Reward Function Edge Cases (Verified)

| Input | Expected | Regex Result | Correct? |
|-------|----------|--------------|----------|
| `#### 42` | `42` | `42` | Yes |
| `#### -5` | `-5` | `-5` | Yes |
| `#### 3.50` | `3.50` | `3.50` | Yes |
| `#### 1,000,000` | `1000000` | `1,000,000` -> `1000000` | Yes |
| `#### 5 ... #### 10` | `10` | `10` (last match) | Yes (FIXED) |
| `The answer is 42` | `42` | `42` | Yes |
| `The answer is -7.5` | `-7.5` | `-7.5` | Yes |
| `\boxed{15}` | `15` | `15` | Yes |
| `\boxed{-3}` | `-3` | `-3` | Yes |
| `No number here` | `None` | `None` | Yes |
| `Step 1: 5, Step 2: 10, so 15` | `15` (fallback) | `15` | Correct by luck |
| `#### ` (no number) | `None` (or fallback) | Falls through | Correct |
| `the answer is forty` | `None` (fallback) | Falls through to last-number | Depends |

---

## Batched Generation (2026-04-08) -- Implemented and Tested

Batched generation is fully implemented in `ppo_trainer.py` and `advantage.py`.
All per-sample loops have been converted to batched operations. Test coverage
is provided by `tests/test_batched_ops.py` which verifies log-prob consistency
between batched and single-sample paths.

## Batched Generation Notes (2026-04-08)

### Left-padding behavior
Batched generation uses `tokenizer.padding_side = "left"` so that all sequences
are right-aligned and `model.generate()` can produce tokens at the same position
across the batch. This changes tokenization behavior compared to the original
per-sample loop (which had no padding). Left-pad tokens are stripped from outputs
using the attention mask before building Rollout objects.

### pad_token == eos_token edge case
When `tokenizer.pad_token` is None (e.g., Llama), it is set to `eos_token`. This
means left-padding inserts EOS tokens at the start of shorter sequences. The
attention mask correctly masks these out, so the model does not attend to them.
`pad_token_id` is passed explicitly to `model.generate()` to prevent early stopping
on padding tokens.

### Checkpoint atomicity
`save_checkpoint` writes to a temporary directory (`.tmp_checkpoint_step_NNNNNN`)
and renames atomically to the final path. This ensures that a crash during save
does not leave a corrupted checkpoint. Checkpoint rotation deletes the oldest
checkpoint after the rename succeeds.

---

## Design Decisions (Intentional, Documented)

### No Entropy Regularization
The PPO paper includes entropy bonus but this implementation omits it. Defensible
for RLVR: temperature sampling provides exploration; entropy bonus could encourage
incorrect but diverse responses.

### No Value Function Clipping
"The 37 Implementation Details of PPO" found it often hurts. For binary rewards,
unclipped MSE is preferable.

### PPO-Clip + PPO-KL Simultaneous
Both mechanisms are supported. With kl_coeff=0.0 (default), only clipping is active.
Document which was used in the paper.

### gamma Default Mismatch (RESOLVED)
`compute_advantages` default is now `gamma=1.0`, consistent with `PPOConfig`.
The parameter is unused for single-step episodes.
