# Logic Fix Spec

This spec covers correctness of the PPO algorithm and experimental measurement logic.
Items marked ✓ PASS were reviewed and found correct; no change is needed.
Items marked ✗ BUG require fixes before running experiments.

---

## Confirmed Correct (no change needed)

| Item | Location | Status |
|------|----------|--------|
| PPO-clip surrogate `min(ρA, clip(ρ,1±ε)A)` | `ppo_trainer.py:259–263` | ✓ PASS |
| Advantages detached before ratio computation | `ppo_trainer.py:251` | ✓ PASS |
| `old_log_prob` stored at generation time (before update) | `ppo_trainer.py:164–167` | ✓ PASS |
| Critic MSE loss against observed returns | `ppo_trainer.py:348` | ✓ PASS |
| Critic hidden states detached from policy graph | `ppo_trainer.py:338–343` | ✓ PASS |
| GAE = `r − V(s)` for single-step terminal-reward episodes | `advantage.py:37–50` | ✓ PASS |
| MC baseline = `E[r | prompt]` = mean reward over samples | `advantage.py:56–118` | ✓ PASS |
| Advantage normalization guard for batch size 1 | `advantage.py:47` | ✓ PASS |

---

## L1 — `n_ppo_epochs` defined but never used [CRITICAL]

**Location:** `ppo_specs/config.py:23` (defined), `ppo_specs/ppo_trainer.py` (`train_step`, never referenced)

**Problem:**
```python
# config.py
n_ppo_epochs: int = 1   # set but ignored

# ppo_trainer.py
def train_step(self, prompts, ground_truths):
    batch = self.generate_rollouts(prompts, ground_truths)
    metrics = self.ppo_update(batch)   # always exactly 1 update
    ...
```
Standard PPO performs **K gradient updates on the same collected batch** before
collecting a new one (Schulman et al., 2017). The current code performs exactly 1
regardless of `n_ppo_epochs`, making the config parameter misleading and the
algorithm non-standard.

**Why it matters:** With `n_ppo_epochs=1` the difference is zero, but the
experimental design may want to compare `n_ppo_epochs ∈ {1, 4}`. More importantly,
reporting that "PPO was run" implies the standard algorithm.

**Fix:**

In `ppo_trainer.py`, update `train_step`:
```python
def train_step(
    self,
    prompts: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """Single PPO iteration: collect rollouts → K gradient updates."""
    batch = self.generate_rollouts(prompts, ground_truths)

    all_metrics: list[dict[str, float]] = []
    for _ in range(self.config.n_ppo_epochs):
        metrics = self.ppo_update(batch)
        all_metrics.append(metrics)

    # Average scalar metrics over epochs; keep final clip_fraction
    aggregated = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }
    aggregated["accuracy"] = compute_accuracy(
        [r.reward for r in batch.rollouts]
    )
    aggregated["total_rollouts"] = self.total_rollouts
    self.step += 1
    return aggregated
```

---

## L2 — `kl_coeff` defined but never applied [CRITICAL]

**Location:** `ppo_specs/config.py:24` (defined), `ppo_specs/ppo_trainer.py:266` (loss computation, never used)

**Problem:**
```python
# config.py
kl_coeff: float = 0.0   # KL penalty coefficient — ignored

# ppo_trainer.py
total_loss = policy_loss + 0.5 * critic_loss   # no KL term
```
The default `kl_coeff=0.0` means training is functionally correct with the current
code, but the feature cannot be enabled. KL penalty against the reference policy is a
common PPO stabilisation technique in RLHF/RLVR.

**Fix:**

In `ppo_trainer.py`, inside `ppo_update`, after `new_log_probs` is computed:
```python
# Optional KL penalty: KL(π_old || π_new) ≈ old_log_p − new_log_p
# (first-order approximation; add to loss to penalise large policy updates)
kl_penalty = torch.tensor(0.0, device=self.device)
if self.config.kl_coeff > 0.0:
    kl_penalty = self.config.kl_coeff * (
        old_log_probs.detach() - new_log_probs
    ).mean()

total_loss = policy_loss + self.config.critic_loss_weight * critic_loss + kl_penalty
```
Log `kl_penalty.item()` in the returned metrics dict for monitoring.

---

## L3 — Advantage error metric wrong for trainable critics in E2.7 [CRITICAL]

**Location:** `ppo_specs/run_e2_7.py:124–129`

**Problem:**
```python
est = np.full(len(mc_baselines), metrics["mean_reward"])
mc  = np.array(list(mc_baselines.values()))
adv_error = advantage_estimation_error(est, mc)
```
`metrics["mean_reward"]` is the batch-mean reward — the REINFORCE baseline. This is
correct for `critic_capacity="none"`, but for `"small"`, `"medium"`, and `"large"`
critics the estimated baseline should be the critic's actual output `V̂(s)` on the
reference prompts.

**Why it matters:** E2.7 measurement (iv) — |Â − A_MC| — is supposed to show that
PPO's critic introduces bias relative to the MC ground truth. Using batch-mean for all
critics makes the error look identical regardless of critic quality.

**Fix:**

1. Add `_eval_critic_on_prompts` to `PPOTrainer` (see below).
2. In `run_e2_7.py`, update the advantage error block:

```python
if mc_baselines:
    ref_prompts = list(mc_baselines.keys())
    mc_vals = np.array(list(mc_baselines.values()))

    if config.critic_capacity == "none":
        # REINFORCE: batch mean is the baseline
        est_vals = np.full(len(mc_vals), metrics["mean_reward"])
    else:
        # Trainable critic: evaluate V̂(s) on reference prompts
        est_vals = trainer._eval_critic_on_prompts(ref_prompts)

    adv_error = advantage_estimation_error(est_vals, mc_vals)
    log_entry["advantage_error"] = adv_error
```

---

## L4 — Critic error εV wrong for trainable critics in E2.8 [CRITICAL]

**Location:** `ppo_specs/run_e2_8.py:122–129`

**Problem:**
```python
if capacity == "none":
    est_vals = np.full(len(mc_vals), metrics["mean_reward"])
else:
    # BUG: identical to the "none" branch
    est_vals = np.full(len(mc_vals), metrics["mean_reward"])
```
Both branches compute the same value. The comment in the else-branch acknowledges
this is a placeholder but marks it as acceptable ("cheaper proxy"), which is
incorrect — using batch mean for all capacities makes `εV` identical across all
critics, rendering E2.8 meaningless.

**Fix:**

Same `_eval_critic_on_prompts` method. In `run_e2_8.py`:
```python
if capacity == "none":
    est_vals = np.full(len(mc_vals), metrics["mean_reward"])
else:
    est_vals = trainer._eval_critic_on_prompts(ref_prompts)

ev   = critic_approximation_error(est_vals, mc_vals)
bias = advantage_estimation_error(est_vals, mc_vals)
```
Also pass `ref_prompts = list(mc_baselines.keys())` into `run_one_capacity` so it
is available inside the eval block.

---

## Required New Method: `_eval_critic_on_prompts`

Add to `PPOTrainer` in `ppo_trainer.py`:

```python
@torch.no_grad()
def _eval_critic_on_prompts(self, prompts: list[str]) -> np.ndarray:
    """
    Evaluate the trained critic V̂(s) on a list of prompts.

    Used for E2.7/E2.8 advantage estimation error and εV measurements.
    Requires a trainable critic (capacity != "none"); raises ValueError otherwise.

    Args:
        prompts: Prompt strings to evaluate

    Returns:
        numpy array of shape [len(prompts)] with critic value estimates
    """
    if not self.critic.is_trainable():
        raise ValueError(
            "_eval_critic_on_prompts requires a trainable critic; "
            "use batch-mean reward for capacity='none'."
        )
    self.model.eval()
    self.critic.eval()
    values: list[float] = []

    for prompt in prompts:
        enc = self._tokenize_prompt(prompt)   # uses the helper from readability R3
        outputs = self.model(
            input_ids=enc["input_ids"],
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]   # [1, H]
        v = self.critic(last_hidden).squeeze().item()
        values.append(v)

    return np.array(values, dtype=np.float32)
```

---

## L5 — `gamma` parameter unused (by design) — add clarifying comment [INFO]

**Location:** `ppo_specs/advantage.py:22–24`

**Problem:** `gamma` is a parameter of `compute_advantages` but is not used in the
body. This is intentional (single-step episodes make γ irrelevant) but surprising to
anyone reading the function signature.

**Fix:** Add a one-line inline comment:
```python
def compute_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor],
    gamma: float = 0.99,    # retained for API compatibility; unused for single-step
    normalize: bool = True,
) -> torch.Tensor:
```
