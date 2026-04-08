# Readability Fix Spec

Each issue follows the format: **Problem → Why it matters → Step-by-step fix**.

---

## R1 — `_cycle_batch` duplicated in both run scripts

**Status**: **Fixed** (2026-04-08)

**Location:** `run_e2_7.py`, `run_e2_8.py`

**Problem:** Identical helper function defined twice.

**Resolution:** Both scripts now import `cycle_batch` from `ppo_specs.utils`
instead of defining local copies.

---

## R2 — MC baseline setup duplicated

**Location:** `run_e2_7.py:77–93`, `run_e2_8.py:184–204`

**Problem:** Both scripts contain nearly identical blocks that pick `n_mc`, slice 5 reference prompts, call `estimate_mc_advantages`, and print results.

**Why it matters:** The two blocks will drift. The E2.8 version already uses a temporary trainer just for this; the E2.7 version reuses the main trainer. Consolidating forces the caller to pass in the right trainer.

**Fix:**
1. `ppo_specs/utils.py` already defines `setup_mc_baselines(trainer, train_prompts, train_gts, n_steps, max_new_tokens, device)`.
2. In `run_e2_7.py`, replace lines 77–93 with:
   ```python
   from ppo_specs.utils import setup_mc_baselines
   mc_baselines = setup_mc_baselines(
       trainer, train_prompts, train_gts,
       config.n_steps, config.max_new_tokens, device,
   ) if compute_mc else {}
   ```
3. In `run_e2_8.py`, replace lines 184–204 with:
   ```python
   from ppo_specs.utils import setup_mc_baselines
   mc_baselines = setup_mc_baselines(
       tmp_trainer, train_prompts, train_gts,
       config.n_steps, config.max_new_tokens, device,
   )
   ```

---

## R3 — Tokenization pattern repeated 4× with hardcoded `max_length=512`

**Location:** `ppo_trainer.py:144`, `ppo_trainer.py:332`, `ppo_trainer.py:385`, `advantage.py:97`

**Problem:** Every call to the tokenizer spells out the same four kwargs. The `max_length=512` literal appears four times with no connection to any config value.

**Why it matters:** Changing the max prompt length requires hunting down four sites. A missed update silently truncates inputs differently across code paths.

**Fix:**
1. Add to `PPOConfig`:
   ```python
   tokenize_max_length: int = 512
   ```
2. Add a private helper to `PPOTrainer`:
   ```python
   def _tokenize_prompt(self, prompt: str) -> dict:
       return self.tokenizer(
           prompt,
           return_tensors="pt",
           truncation=True,
           max_length=self.config.tokenize_max_length,
           padding=False,
       ).to(self.device)
   ```
3. Replace all four tokenizer call sites with `self._tokenize_prompt(prompt)`.
4. In `advantage.py`, `estimate_mc_advantages` already accepts `max_new_tokens`; add `max_prompt_length: int = 512` parameter and thread it through the tokenizer call.

---

## R4 — `advantage_estimation_error` defined in two places

**Location:** `ppo_specs/advantage.py:123`, `eval/metrics.py:29`

**Problem:** Two functions with the same name computing the same quantity. They are not in sync — `advantage.py` uses MAE, `eval/metrics.py` also uses MAE but operates on `(estimated, mc_advantages)` with different variable names.

**Why it matters:** Any caller that imports from the wrong module gets a subtly different API. Future changes to the metric will need to be applied twice.

**Fix:**
1. Keep `advantage_estimation_error` in `ppo_specs/advantage.py` as the canonical version (already more complete with `critic_approximation_error` sibling).
2. In `eval/metrics.py`, replace the function body with a re-export:
   ```python
   from ppo_specs.advantage import advantage_estimation_error  # canonical
   ```
   Or, if the import direction must stay one-way, delete the duplicate and update all callers.

---

## R5 — Magic numbers scattered across files

**Location:** `ppo_trainer.py:266` (`0.5`), `ppo_trainer.py:274` (`1.0`), `advantage.py:48` (`1e-8`), `advantage.py:108` (`0.7`), `run_e2_7.py:82–85` (`10`, `50`, `5`)

**Problem:** Unnamed constants whose meaning must be inferred from context.

**Why it matters:** `0.5` for critic loss weight is a common hyperparameter; researchers will want to sweep it. `1e-8` is an epsilon that should match across normalization calls.

**Fix:**
1. Add to `PPOConfig`:
   ```python
   critic_loss_weight: float = 0.5    # λ_V in total_loss = L_policy + λ_V * L_critic
   grad_clip_norm: float = 1.0        # max gradient norm for policy update
   adv_norm_eps: float = 1e-8         # epsilon for advantage normalization denominator
   mc_samples_local: int = 10         # MC rollouts per prompt in local/dev runs
   mc_samples_cluster: int = 50       # MC rollouts per prompt in full runs (use 1000 on cluster)
   mc_ref_n_prompts: int = 5          # number of reference prompts for MC estimation
   ```
2. In `ppo_trainer.py:266`: `total_loss = policy_loss + self.config.critic_loss_weight * critic_loss`
3. In `ppo_trainer.py:274`: `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)`
4. In `advantage.py`: pass `eps: float = 1e-8` as a parameter, default `1e-8`; call sites pass `self.config.adv_norm_eps`.
5. Add `temperature: float = 0.7` parameter to `estimate_mc_advantages`; callers pass `config.temperature`.
6. In `utils.py`, `setup_mc_baselines` already reads `n_steps` to decide sample count; thread `config.mc_samples_local` / `config.mc_samples_cluster` through.

---

## R6 — Misleading variable names in `run_e2_7.py`

**Location:** `run_e2_7.py:97` (`reward_window`), `run_e2_7.py:121` (`stability`)

**Problem:**
- `reward_window` is not a fixed-size window — it accumulates forever and is sliced at read time.
- `stability` is the training metric name from the spec; the variable should match.

**Fix:**
1. Rename `reward_window` → `reward_history` everywhere in `run_e2_7.py`.
2. Rename `stability` → `reward_variance` everywhere in `run_e2_7.py`.
3. Update the log_entry key to `"reward_variance"` (currently already correct in the dict).

---

## R7 — Identical if/else branches in `run_e2_8.py`

**Status**: **Fixed** (2026-04-08)

**Location:** `run_e2_8.py`

**Problem:** Both if/else branches assigned the same value (`np.full(len(mc_vals), metrics["mean_reward"])`).

**Resolution:** The branches are now genuinely distinct. The `"none"` branch generates
rollouts on reference prompts and uses their batch mean reward. The `else` branch calls
`trainer._eval_critic_on_prompts(ref_prompts_for_eval)` for actual critic values.
See also L3/L4 in `logic.md`.

---

## R8 — Dead intermediate variable `token_lp`

**Location:** `ppo_trainer.py` inside `_policy_log_probs` and `_sequence_log_prob`

**Problem:**
```python
token_lp = response_log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
return token_lp.sum(dim=-1)
```
`token_lp` is assigned and immediately consumed in the next line. The variable name is also an abbreviation.

**Fix:**
Inline into a single expression:
```python
return (
    response_log_probs
    .gather(2, response_ids.unsqueeze(-1))
    .squeeze(-1)
    .sum(dim=-1)
)
```

---

## R9 — Inconsistent type annotation style

**Location:** `ppo_trainer.py` (uses `List[T]`, `Dict[K,V]` from `typing`), `run_e2_7.py:97` and `run_e2_8.py` (uses `list[T]` modern style)

**Problem:** Mixed use of `typing.List` / `typing.Dict` (Python 3.5–3.8 style) and built-in `list[T]` / `dict[K,V]` (Python 3.9+ style) within the same package.

**Fix:**
1. Remove `from typing import List, Dict, Optional, Tuple` from `ppo_trainer.py` and `advantage.py`.
2. Replace all `List[X]` → `list[X]`, `Dict[K,V]` → `dict[K,V]`, `Tuple[A,B]` → `tuple[A,B]`.
3. Keep `Optional[X]` → `X | None` (Python 3.10+) or keep `Optional` from typing — either is acceptable; pick one.
4. Verify Python version in `requirements.txt` / CI is >= 3.9.

---

## R10 — `ALL_CAPACITIES` defined in wrong file

**Status**: Partially Fixed (2026-04-08)

**Location:** `ppo_specs/config.py` (line 72), `ppo_specs/run_e2_8.py` (line 62)

**Problem:** The list of valid critic capacities was buried inside `run_e2_8.py`.

**Current state:** `CRITIC_CAPACITIES: list[str] = ["none", "small", "medium", "large"]`
is now defined in `config.py`. However, `run_e2_8.py` still defines a local
`ALL_CAPACITIES` instead of importing from `config.py`. The import should be added
and the local definition removed.

---

## R11 — `format_prompt_with_template` exists but is never used

**Location:** `src/data.py:56–79`

**Problem:** `format_prompt_with_template()` correctly handles chat templates for
instruction-tuned models (Qwen, Llama), but the training pipeline exclusively calls
`format_prompt()` which uses plain text formatting. This means the more sophisticated
function was written but never integrated.

**Why it matters:** Using plain text prompts with instruction-tuned models (e.g.,
Qwen2.5-0.5B-**Instruct**) is a potential confound. The model may perform worse
because inputs don't match its training format. A reader would not realize this
unless they trace through both data loading paths. See `logic.md` L12 for the
correctness angle.

**Fix:**
1. In `run_e2_7.py` and `run_e2_8.py`, after loading the trainer, format prompts
   using the tokenizer:
   ```python
   from src.data import format_prompt_with_template
   train_prompts = [format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in train_ds]
   ```
2. Or, if plain text is intentional for research consistency, add a comment
   explaining the design choice.

---

## R12 — Missing class-level docstring on `PPOConfig`

**Location:** `ppo_specs/config.py:13`

**Problem:** The `PPOConfig` dataclass has no top-level docstring. A reader must scan all 40 fields to understand what it configures.

**Fix:**
Add directly under `@dataclass`:
```python
@dataclass
class PPOConfig:
    """
    Unified configuration for PPO training on RLVR tasks.

    Field groups
    ────────────
    Model          model_name
    PPO            learning_rate, critic_lr, clip_epsilon, gamma,
                   n_ppo_epochs, kl_coeff, critic_loss_weight, grad_clip_norm
    Rollout        n_rollouts_per_prompt, batch_size, max_new_tokens,
                   temperature, do_sample, tokenize_max_length
    Critic (E2.8)  critic_capacity  ("none" | "small" | "medium" | "large")
    Schedule       n_steps, eval_every, log_every
    Data           n_train_samples, seed
    MC estimation  mc_samples_local, mc_samples_cluster, mc_ref_n_prompts, adv_norm_eps
    Bookkeeping    experiment_name, output_dir
    """
```
