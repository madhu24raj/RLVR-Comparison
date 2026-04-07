# Readability Fix Spec

Each issue follows the format: **Problem → Why it matters → Step-by-step fix**.

---

## R1 — `_cycle_batch` duplicated in both run scripts

**Location:** `run_e2_7.py:48–55`, `run_e2_8.py:64–69`

**Problem:** Identical helper function defined twice. Any future change (e.g. adding a shuffle option) must be applied in two places.

**Why it matters:** Silent divergence — one copy gets updated, the other doesn't, and the bug only shows up in one experiment.

**Fix:**
1. Open `ppo_specs/utils.py` — `cycle_batch` is already defined there.
2. In `run_e2_7.py`, delete lines 48–55 and add at the top:
   ```python
   from ppo_specs.utils import cycle_batch
   ```
3. Repeat for `run_e2_8.py` lines 64–69.
4. Replace all calls `_cycle_batch(...)` → `cycle_batch(...)`.

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

**Location:** `run_e2_8.py:122–129`

**Problem:**
```python
if capacity == "none":
    est_vals = np.full(len(mc_vals), metrics["mean_reward"])
else:
    # Use mean reward as a proxy for critic output
    est_vals = np.full(len(mc_vals), metrics["mean_reward"])
```
Both branches assign the exact same value. The else-branch comment admits this is a placeholder but doesn't fix it.

**Why it matters:** This is also a logic bug (see `logic.md` L4). For readability, the dead branch confuses readers into thinking the two cases differ.

**Fix:**
See `logic.md` L4 for the correct fix (evaluate actual critic values for trainable critics). Once fixed, the if/else will be genuinely distinct and the comment can be removed.

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

**Location:** `run_e2_8.py:59`

**Problem:** The list of valid critic capacities is a config-level constant but is buried inside a run script, making it invisible to anything that imports `config.py`.

**Fix:**
1. In `ppo_specs/config.py`, add after the `PPOConfig` dataclass:
   ```python
   CRITIC_CAPACITIES: list[str] = ["none", "small", "medium", "large"]
   ```
2. In `run_e2_8.py`, replace:
   ```python
   ALL_CAPACITIES = ["none", "small", "medium", "large"]
   ```
   with:
   ```python
   from ppo_specs.config import CRITIC_CAPACITIES
   ```
3. Update all references `ALL_CAPACITIES` → `CRITIC_CAPACITIES` in `run_e2_8.py`.

---

## R11 — Missing class-level docstring on `PPOConfig`

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
