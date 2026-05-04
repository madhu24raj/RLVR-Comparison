# Phase 2 Integration Beads

Companion to [reward_model_integration.md](reward_model_integration.md),
[reward_model_agent_prompts.md](reward_model_agent_prompts.md),
[ddp_cpu_gpu_migration.md](ddp_cpu_gpu_migration.md), and
[ddp_agent_prompts.md](ddp_agent_prompts.md).

This document defines the unit-of-work breakdown ("beads") for Phase 2
of the two integration efforts in this repo: the learned **R**eward
**M**odel and the **D**istributed **D**ata **P**arallel migration.
Each bead is a self-contained 30–90-minute unit assignable to one
sub-agent.

## Status snapshot (2026-05-04)

| Track | Status | Remaining beads |
|-------|--------|-----------------|
| RM    | NOT STARTED | RM-A, RM-B, RM-TEST (+ optional RM-D1, RM-D2, RM-D3, RM-D4) |
| DDP   | PARTIAL — trainer + checkpoint + run scripts done | DDP-A, DDP-E, DDP-BAR, DDP-CLEAN (low priority), DDP-F1 (smoke), DDP-F2 (review) |

The original Agent A in `ddp_agent_prompts.md` did two things; the
`configs/accelerate_cpu.yaml` half is done. Only the `setup_env.sh
--cpu-only` half is left, captured here as **DDP-A**.

## Baseline invariant (locked, applies to every bead)

> The PPO trainer is, at its most baseline, a PPO trainer with custom
> reward functions. The existing `reward_fn` callable interface, the
> `reward_mode` knob, `_RewardFnWrapper`, `make_reward_fn`, and
> `SelfJudgeRewardModel` all stay. The new `RewardModelScorer` is one
> more custom reward function that plugs into the same `reward_fn`
> slot. When `reward_model_capacity == "none"`, the trainer behavior is
> bit-identical to today's deterministic / self_judge / combined paths.

Every bead's "Invariants to preserve" section restates the slice of
this invariant that applies to it.

## Bead key

- **Files to read** lists the *ground truth* the agent needs (current
  line numbers as of 2026-05-04).
- **Files to edit** lists the surgical insertion sites.
- **Hard preconditions** are other beads that MUST land first.
- **Self-verification** are commands the agent runs before claiming done.

---

## RM-A — Implement `ppo_specs/reward_model.py` + config + checkpoint hash

**Owner:** general-purpose sub-agent. **Estimated time:** ~75 min.

**What this delivers:** the standalone learned-RM module (capacity tiers,
factory, blend, frozen-weight invariants), the 6 new `PPOConfig` fields,
and the 2 missing `_config_hash` entries.

**Files to read:**
- [ppo_specs/specs/reward_model_integration.md](reward_model_integration.md) — full spec, especially the "Baseline invariant", "Interaction with `reward_mode`", "Adapter contract", and "Checkpoint hash invariant" sections.
- [ppo_specs/critic.py:106-131](../critic.py#L106-L131) — the `build_critic` factory pattern to mirror.
- [ppo_specs/ppo_trainer.py:1040-1197](../ppo_trainer.py#L1040-L1197) — `load_ppo_trainer` reference for the model load pattern (especially the `accelerator.main_process_first` cache-warming at L1079-L1099 / L1152-L1158).
- [ppo_specs/ppo_trainer.py:514-571](../ppo_trainer.py#L514-L571) — `_extract_last_hidden`, the batched-LM-forward + last-real-token-hidden-state pattern.
- [shared/per_token_loss.py:21-](../../shared/per_token_loss.py#L21) — `batched_per_token_log_probs` reusable helper.
- [src/rewards.py:84-105](../../src/rewards.py#L84-L105) — `gsm8k_reward` (the `none`-tier wraps this).
- [src/rewards.py:201-285](../../src/rewards.py#L201-L285) — existing `_RewardFnWrapper` and `make_reward_fn`. **Read carefully — DO NOT delete or rewrite per the Baseline invariant.**

**Files to edit:**
- **NEW** `ppo_specs/reward_model.py` — `RewardModelScorer`, `NoneReward`, `LearnedRMScorer`, `BlendedScorer`, `build_reward_model(config, device, *, base_model=None)`. Mirror critic.py's structure: classes, then a single factory function, then a `__main__` smoke that exercises the `none` tier only (no HF download).
- [ppo_specs/config.py:60](../config.py#L60) — add the 6 fields after the existing reward block (`reward_mode`/`self_judge_weight`/`self_judge_normalize`):
  ```python
  reward_model_capacity: str = "none"
  reward_model_name: str | None = None
  reward_model_dtype: str = "auto"
  reward_model_reuse_reference: bool = False
  reward_blend_alpha: float = 1.0
  reward_score_activation: str = "sigmoid"
  ```
- [ppo_specs/checkpoint.py:213-228](../checkpoint.py#L213-L228) — three of five RM fields are already wired via `getattr` defaults (L223-L225). Add the two missing entries to the `key_fields` dict:
  ```python
  "reward_model_name": getattr(config, "reward_model_name", None),
  "reward_model_reuse_reference": getattr(config, "reward_model_reuse_reference", False),
  ```

**Hard preconditions:** none. RM-A can land in parallel with DDP-A and DDP-E.

**Invariants to preserve:**
- Baseline invariant — the new module is additive. DO NOT touch `src/rewards.py`, `make_reward_fn`, or `_RewardFnWrapper`.
- All RM parameters frozen (`requires_grad_(False)`) at load time.
- `__main__` smoke does NOT touch the network (no HF download). It only exercises the `none` tier on a hand-rolled batch.
- New config fields default to "off" semantics: `reward_model_capacity="none"`, `reward_blend_alpha=1.0`, `reward_score_activation="sigmoid"` reproduces today's behavior bit-identically.

**Self-verification:**
```bash
python -c "from ppo_specs.reward_model import build_reward_model; print('ok')"
python ppo_specs/reward_model.py     # exercises the __main__ none-tier smoke
python -c "from ppo_specs.config import PPOConfig; c = PPOConfig(); print(c.reward_model_capacity, c.reward_blend_alpha)"
python -c "from ppo_specs.checkpoint import _config_hash; from ppo_specs.config import PPOConfig; print(_config_hash(PPOConfig()))"  # must not raise
```
And, by inspection: `grep 'reward_model' ppo_specs/checkpoint.py` shows
five `key_fields` entries (capacity, name, blend_alpha, score_activation,
reuse_reference).

**Reporting contract:** the agent reports the three files touched, a
one-line confirmation that NoneReward output is float-identical to
`gsm8k_reward` on a 3-item hand-rolled batch, and any deviation from
the spec.

---

## RM-B — Wire the RM into PPOTrainer + load_ppo_trainer

**Owner:** general-purpose sub-agent. **Estimated time:** ~60 min.

**What this delivers:** the trainer learns to *prefer* batched
`score_batch(...)` calls when a learned RM is configured, while keeping
the existing per-sample `reward_fn` path live for the `none`-tier and
for the existing `reward_mode` plumbing.

**Files to read:**
- `ppo_specs/reward_model.py` (after RM-A lands).
- [ppo_specs/specs/reward_model_integration.md "Adapter contract"](reward_model_integration.md#adapter-contract) — the exact protocol Agent B must implement.
- [ppo_specs/specs/reward_model_integration.md "Interaction with reward_mode"](reward_model_integration.md#interaction-with-reward_mode-orthogonality-contract) — for the conflict-resolution rule.
- [ppo_specs/ppo_trainer.py:154-244](../ppo_trainer.py#L154-L244) — `__init__` (where the new `reward_model_scorer` arg goes).
- [ppo_specs/ppo_trainer.py:267-389](../ppo_trainer.py#L267-L389) — `generate_rollouts`. The per-sample reward call is at [L343](../ppo_trainer.py#L343).
- [ppo_specs/ppo_trainer.py:954-1035](../ppo_trainer.py#L954-L1035) — `evaluate`. The per-sample reward call is at [L1024](../ppo_trainer.py#L1024).
- [ppo_specs/ppo_trainer.py:1040-1197](../ppo_trainer.py#L1040-L1197) — `load_ppo_trainer`. The `make_reward_fn` call is at [L1173-L1175](../ppo_trainer.py#L1173-L1175).

**Files to edit (one file):** `ppo_specs/ppo_trainer.py` only.
1. **`__init__` ([L154-L244](../ppo_trainer.py#L154-L244)):** add an optional kwarg `reward_model_scorer: Optional[RewardModelScorer] = None`. Store on `self`. Document in the docstring that `reward_fn` is still the primary path; `reward_model_scorer`, when set, takes precedence inside `generate_rollouts` and `evaluate`.
2. **`generate_rollouts` ([L267-L389](../ppo_trainer.py#L267-L389)):** at the per-sample loop ([L333-L361](../ppo_trainer.py#L333-L361)), add a fast path: if `self.reward_model_scorer is not None`, replace the per-sample `self.reward_fn(completion, local_gts[i])` call at [L343](../ppo_trainer.py#L343) with one batched `self.reward_model_scorer.score_batch(local_prompts, completions_so_far, local_gts)` call. Iterate the resulting tensor to populate `r.reward`. Preserve the parse_success / format_match diagnostics — those are string-level and don't depend on the RM.
3. **`evaluate` ([L954-L1035](../ppo_trainer.py#L954-L1035)):** mirror the substitution at [L1024](../ppo_trainer.py#L1024). For the returned `accuracy`, always compute `gsm8k_reward` separately so the metric stays binary regardless of which scorer trained the policy.
4. **`load_ppo_trainer` ([L1040-L1197](../ppo_trainer.py#L1040-L1197)):** after the `make_reward_fn` call at [L1173-L1175](../ppo_trainer.py#L1173-L1175), build the RM scorer:
   ```python
   reward_model_scorer = build_reward_model(
       config, device,
       base_model=reference_model if config.reward_model_reuse_reference else None,
   )
   ```
   Pass to PPOTrainer at [L1177-L1196](../ppo_trainer.py#L1177-L1196).
5. **Conflict-resolution check** (early in `load_ppo_trainer`, before any HF load): if `config.reward_model_capacity != "none"` AND `config.reward_mode != "deterministic"`, raise `ValueError` per the spec. This prevents accidentally combining self_judge with a learned RM.

**Hard preconditions:** RM-A.

**Invariants to preserve:**
- Baseline invariant — when `reward_model_capacity == "none"`, the per-sample loop at L343 / L1024 must execute exactly as today, producing bit-identical reward floats.
- `accuracy` returned by `evaluate` must come from `gsm8k_reward` regardless.
- The conflict-resolution `ValueError` is part of the design, not a bug — do NOT silently allow `self_judge` × non-`none` capacity.
- Frozen-weight check: after `build_reward_model` returns, assert `all(not p.requires_grad for p in reward_model_scorer.parameters())`.

**Self-verification:**
```bash
python -c "from ppo_specs.ppo_trainer import load_ppo_trainer, PPOTrainer; print('ok')"
python ppo_specs/run_e2_7.py --local-test --no-mc   # baseline (capacity=none) must produce bit-identical metrics to pre-change run on the same seed
```
- Source-level: `grep -n 'self.reward_model_scorer' ppo_specs/ppo_trainer.py` shows ≥2 hits (in `__init__` and at least one of `generate_rollouts`/`evaluate`).
- Source-level: `grep -n 'self.reward_fn' ppo_specs/ppo_trainer.py` still shows the L343 and L1024 fallbacks (kept for `none`-tier path).

**Reporting contract:** line ranges touched, baseline parity test result (final_acc match between pre-change and post-change `--local-test --no-mc`), and any signature changes that ripple beyond the four sites listed.

---

## RM-TEST — Unit tests for the new RM module

**Owner:** general-purpose sub-agent. **Estimated time:** ~45 min.

**What this delivers:** `ppo_specs/tests/test_reward_model.py` with the
six tests from the spec. CPU-only, monkeypatched HF loads, total runtime
under 10 s.

**Files to read:**
- [ppo_specs/specs/reward_model_integration.md "Unit tests"](reward_model_integration.md#unit-tests-ppo_specstestsest_reward_modelpy)
- `ppo_specs/reward_model.py` (after RM-A lands).
- [ppo_specs/tests/test_data_rewards.py](../tests/test_data_rewards.py) — for the existing reward-test style.
- [ppo_specs/tests/test_trainer.py](../tests/test_trainer.py) — for the existing PPOTrainer integration-test style (mock model + mock tokenizer).

**Files to edit:**
- **NEW** `ppo_specs/tests/test_reward_model.py` — six tests:
  1. `test_none_tier_parity` — float-exact match with `gsm8k_reward` on a 3-item hand-rolled batch.
  2. `test_small_tier_shape_dtype` — monkeypatch `transformers.AutoModelForCausalLM.from_pretrained`; assert `score_batch` returns `[B]` float32 CPU tensor.
  3. `test_blend_alpha_interpolation` — alpha ∈ {0, 0.5, 1}; mocked learned scorer returns fixed tensor; check convex combination, tol 1e-6.
  4. `test_reuse_reference_weight_sharing` — pass a shared base, assert `id()`-equality on the base parameters.
  5. `test_ground_truth_optional` — mocked learned tier accepts `ground_truths=None`; `none` tier raises `ValueError` with a clear message.
  6. `test_frozen_rm_params` — every parameter has `requires_grad=False` after `build_reward_model(capacity="small")` (mocked).

**Hard preconditions:** RM-A. Can run in parallel with RM-B since the tests
target the standalone module.

**Invariants to preserve:**
- CPU-only. No `torch.cuda.*` calls. Use `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` only if a test genuinely needs GPU — none should.
- No network. All HF loads monkeypatched.
- Total runtime <10 s on a laptop.

**Self-verification:**
```bash
pytest ppo_specs/tests/test_reward_model.py -v        # all 6 pass
pytest ppo_specs/tests/ -v                            # no collateral damage
```

**Reporting contract:** test count, total runtime, any tests adapted from
the spec's plan with reasons.

---

## DDP-A — `setup_env.sh --cpu-only` flag

**Owner:** general-purpose sub-agent. **Estimated time:** ~30 min.

**What this delivers:** `bash scripts/setup_env.sh --cpu-only` installs
the CPU-only torch wheel and skips CUDA verification, while the default
GPU-installing behavior is preserved.

**Files to read:**
- [ppo_specs/specs/ddp_cpu_gpu_migration.md §6](ddp_cpu_gpu_migration.md#6-setup_envsh---cpu-only--not-done-agent-a-still-owes-this) — the exact change list.
- [scripts/setup_env.sh](../../scripts/setup_env.sh) — all 192 lines.

**Files to edit (one file):** `scripts/setup_env.sh`.
- After [L26](../../scripts/setup_env.sh#L26) (`MODEL_CACHE` default), add `CPU_ONLY=false`.
- In the parser at [L29-L39](../../scripts/setup_env.sh#L29-L39), add the case `--cpu-only) CPU_ONLY=true; shift ;;`.
- Step 2 at [L71-L95](../../scripts/setup_env.sh#L71-L95): if `CPU_ONLY=true`, set `TORCH_INDEX="https://download.pytorch.org/whl/cpu"` (bypass the case statement at L76-L82). Skip the CUDA verification block at [L87-L95](../../scripts/setup_env.sh#L87-L95); print only `torch.__version__`.
- Step 4 (model pre-download): unchanged.
- Footer at [L174-L191](../../scripts/setup_env.sh#L174-L191): when `CPU_ONLY=true`, print the smoke command `accelerate launch --config_file configs/accelerate_cpu.yaml ppo_specs/run_e2_7.py --local-test --no-mc`.

**Hard preconditions:** none. Independent of every other bead.

**Invariants to preserve:**
- Default behavior (no flag) installs the CUDA torch wheel exactly as today.
- No Python edits. No SLURM-script edits (DDP-E owns those).

**Self-verification:**
```bash
bash -n scripts/setup_env.sh                                   # syntax-check
bash -x scripts/setup_env.sh --cpu-only --skip-models 2>&1 | grep TORCH_INDEX
   # expected: TORCH_INDEX=https://download.pytorch.org/whl/cpu
bash -x scripts/setup_env.sh --skip-models 2>&1 | grep TORCH_INDEX
   # expected: TORCH_INDEX=https://download.pytorch.org/whl/cu121 (or similar)
```

**Reporting contract:** the file touched, the TORCH_INDEX value observed in
both branches, and any deviation from the spec.

---

## DDP-E — `DEVICE_MODE` branching in slurm scripts

**Owner:** general-purpose sub-agent. **Estimated time:** ~60 min.

**What this delivers:** `scripts/slurm_e2_7.sh` and `scripts/slurm_e2_8.sh`
both branch on `DEVICE_MODE=cpu|gpu` (default `gpu`); CPU mode skips
`module load cuda` and the NCCL exports, picks
`configs/accelerate_cpu.yaml`, and uses `accelerate launch`. The
single-GPU branch is unified to also use `accelerate launch` (currently
it uses bare `python`, which means single-GPU never exercises the
Accelerator code path).

**Files to read:**
- [ppo_specs/specs/ddp_cpu_gpu_migration.md §5](ddp_cpu_gpu_migration.md#5-slurm-parameterization) — the exact change list.
- [scripts/slurm_e2_7.sh](../../scripts/slurm_e2_7.sh) — 158 lines.
- [scripts/slurm_e2_8.sh](../../scripts/slurm_e2_8.sh) — 178 lines.
- `configs/accelerate_cpu.yaml` — already at the repo root.
- `configs/accelerate_multi_gpu.yaml` — for the GPU-default value.

**Files to edit:** both `scripts/slurm_e2_7.sh` and `scripts/slurm_e2_8.sh`.
1. Configurable-parameters block (slurm_e2_7.sh L40-L50; slurm_e2_8.sh L40-L51): add `DEVICE_MODE="${DEVICE_MODE:-gpu}"` and `NUM_PROCESSES="${NUM_PROCESSES:-4}"`.
2. Derived-settings block: add the `if [ "$DEVICE_MODE" = "cpu" ]` branch that picks `ACCEL_CONFIG=configs/accelerate_cpu.yaml` vs `configs/accelerate_multi_gpu.yaml`.
3. Gate `module load cuda/12.1` at slurm_e2_7.sh:72 and slurm_e2_8.sh:75 behind `DEVICE_MODE != cpu`.
4. Gate the NCCL block at slurm_e2_7.sh:86-93 and slurm_e2_8.sh:88-93 behind `DEVICE_MODE != cpu`.
5. Add the runtime sanity warning when `DEVICE_MODE=cpu` but `SLURM_JOB_GRES` mentions `gpu`.
6. Replace the bare-python single-GPU branch at slurm_e2_7.sh:137-142 with `accelerate launch --config_file "$ACCEL_CONFIG" --num_processes "$NUM_PROCESSES" ...`. In slurm_e2_8.sh, replace the per-capacity bare-python at L152-L153 similarly. Keep the `multigpu` and `parallel` modes unchanged in shape.

**Hard preconditions:** none. Independent of DDP-A and the RM beads.

**Invariants to preserve:**
- Default (`DEVICE_MODE=gpu`, no env var override) preserves today's behavior exactly: NCCL exports fire, `module load cuda` runs, the GPU accelerate config is used.
- The `#SBATCH` directives at the top are NOT modified (resource overrides happen at submission time).
- The preemption handler at slurm_e2_7.sh:109-118 / slurm_e2_8.sh:108-117 is untouched.

**Self-verification:**
```bash
bash -n scripts/slurm_e2_7.sh   # syntax-check
bash -n scripts/slurm_e2_8.sh
DEVICE_MODE=cpu bash -x scripts/slurm_e2_7.sh 2>&1 | head -60
   # expected: ACCEL_CONFIG=configs/accelerate_cpu.yaml; no `module load cuda` line
DEVICE_MODE=gpu bash -x scripts/slurm_e2_7.sh 2>&1 | head -60
   # expected: ACCEL_CONFIG=configs/accelerate_multi_gpu.yaml; module load cuda attempted
```

**Reporting contract:** the exact `[RUN] ...` line emitted by each script
under each `DEVICE_MODE`, plus any SLURM_MODE × DEVICE_MODE interactions
that needed reasoning beyond the spec.

---

## DDP-BAR — Audit/strengthen barriers around `save_checkpoint`

**Owner:** general-purpose sub-agent. **Estimated time:** ~30 min.

**What this delivers:** confirmation that every `save_checkpoint` call site
in the run scripts is preceded by `accelerator.wait_for_everyone()` (via
the `_wait()` helper or directly), and a target of **≥4 effective
barriers** in `run_e2_7.py` plus **≥1 barrier** before any future
`save_checkpoint` site in `run_e2_8.py`.

**Files to read:**
- [ppo_specs/specs/ddp_cpu_gpu_migration.md §2.1 "Barriers before every checkpoint save"](ddp_cpu_gpu_migration.md#21-run_e27py--landed) — the contract.
- [ppo_specs/run_e2_7.py:319-360](../run_e2_7.py#L319-L360) — current barrier sites (4 effective).
- [ppo_specs/run_e2_8.py:295-345](../run_e2_8.py#L295-L345) — current barrier sites (no save_checkpoint calls today).
- [ppo_specs/checkpoint.py:50-63](../checkpoint.py#L50-L63) — caller-gated barrier docstring (the contract Agent D landed).

**Files to edit (only if the audit finds gaps):**
- `ppo_specs/run_e2_7.py` — add any missing `_wait()` predecessors.
- `ppo_specs/run_e2_8.py` — when E2.8 adds `save_checkpoint` calls, mirror the run_e2_7 pattern.

**Hard preconditions:** none. (Independent of all other beads.)

**Invariants to preserve:**
- The barrier MUST live in the CALLER, not inside `save_checkpoint`. Putting it inside deadlocks because non-rank-0 ranks early-return.
- Every `save_checkpoint` call MUST be preceded by a barrier reaching every rank. The pattern is: `_wait()` → `if _is_main(): save_checkpoint(...)` → optional trailing `_wait()` for clean shutdown.

**Self-verification:**
```bash
# In run_e2_7.py: every save_checkpoint must have a _wait() within ~10 lines above.
grep -n -B 10 "save_checkpoint(" ppo_specs/run_e2_7.py | grep -E "(_wait|wait_for_everyone|save_checkpoint)"
# Each save_checkpoint should be preceded by a _wait() in the matching block.

# Effective barrier count (count of _wait() calls):
grep -c "_wait()" ppo_specs/run_e2_7.py
# expected: ≥ 4
```

**Reporting contract:** the audit table (each save_checkpoint call site
× whether a `_wait()` precedes it), any gaps fixed, and the final
`grep -c '_wait()'` count.

---

## DDP-CLEAN — `load_ppo_trainer` print rank-0 gating (low priority)

**Owner:** general-purpose sub-agent. **Estimated time:** ~20 min.

**What this delivers:** the four `print(...)` calls inside
`load_ppo_trainer` that currently fire on every rank under multi-process
Accelerate get gated on rank 0.

**Files to read:**
- [ppo_specs/ppo_trainer.py:1040-1197](../ppo_trainer.py#L1040-L1197) — the function.

**Files to edit:** `ppo_specs/ppo_trainer.py`.
- [L1072](../ppo_trainer.py#L1072) — startup print.
- [L1107](../ppo_trainer.py#L1107) — gradient-checkpointing print.
- [L1110-L1111](../ppo_trainer.py#L1110-L1111) — hidden-size print.
- [L1125-L1126](../ppo_trainer.py#L1125-L1126) — reference-model print.

Wrap each with `if accelerator is None or accelerator.is_main_process:`,
or thread `accelerator` through and use a `print0` helper.

**Hard preconditions:** none.

**Invariants to preserve:** Single-process behavior is unchanged (every print still fires).

**Self-verification:** Manual inspection. Optionally, run a 2-process CPU smoke and confirm each print appears exactly once.

**Reporting contract:** the four lines edited; a one-line description of
the gating helper used.

**Priority:** LOW. The duplicates are already temporally serialized
because `accelerator.main_process_first()` runs HF loads serially. Cosmetic
log clutter only.

---

## RM-D1 / DDP-F1 — End-to-end smoke verification

**Owner:** general-purpose sub-agent. **Estimated time:** ~45 min.

**What this delivers:** five-recipe smoke run producing the table from
[ddp_cpu_gpu_migration.md §7](ddp_cpu_gpu_migration.md#7-smoke-test-recipe-runbook),
extended with a `none` tier RM parity check and (when GPU available)
a `small` tier RM smoke.

**Files to read:**
- [ppo_specs/specs/ddp_cpu_gpu_migration.md §7](ddp_cpu_gpu_migration.md#7-smoke-test-recipe-runbook).
- [ppo_specs/specs/reward_model_integration.md §"Verification checklist"](reward_model_integration.md#verification-checklist).

**Files to edit:** none (verification only).

**Hard preconditions:** RM-A, RM-B, RM-TEST, DDP-A, DDP-E, DDP-BAR all landed.

**Invariants to preserve:**
- `none`-tier parity: pre-RM and post-RM `--local-test --no-mc` runs produce identical per-step `mean_reward`, `policy_loss`, `kl_divergence`. Any drift is a regression.
- 1-proc vs 4-proc `final_acc` may differ slightly (per-rank RNG streams diverge during temperature sampling). The PPO loss itself is deterministic on the gathered batch.

**Self-verification:** the table from §7, fully filled.

**Reporting contract:** the table; any run that hung/OOMed/errored with
stdout tail; final_acc deltas; any unexpected files in `git status`.

---

## RM-D3 / DDP-F2 — Final code review

**Owner:** `superpowers:code-reviewer` sub-agent. **Estimated time:** ~30 min.

**What this delivers:** prioritized punch list (Blockers / Non-blocking / Nits) for both tracks against the spec checklists.

**Files to read:**
- All four specs.
- All edits from RM-A, RM-B, RM-TEST, DDP-A, DDP-E, DDP-BAR.

**Files to edit:** none.

**Hard preconditions:** RM-D1 / DDP-F1 has passed.

**Reporting contract:** under 1500 words. Per item: `file:line` + one-sentence description + recommended fix.

---

# Risk and integration analysis

This section enumerates the cross-cutting risks the bead-level prompts
should NOT swallow. These are decisions that affect both tracks or that
need a single answer at integration time.

## R1. `reward_model_capacity` × `reward_mode` — orthogonality vs supersession

**Question:** if a user sets `reward_mode="self_judge"` AND
`reward_model_capacity="small"`, what happens?

**Resolution (locked in
[reward_model_integration.md §"Interaction with reward_mode"](reward_model_integration.md#interaction-with-reward_mode-orthogonality-contract)):**
the two knobs are orthogonal *axes* but not *additively combinable*. The
matrix of valid combinations is:

- `capacity="none"` × any `reward_mode` → today's behavior (deterministic / self_judge / combined).
- `capacity != "none"` × `reward_mode="deterministic"` → RM is the sole training reward; blend with `gsm8k_reward` via `reward_blend_alpha`.
- `capacity != "none"` × `reward_mode != "deterministic"` → **REJECTED at load time** with a clear `ValueError` from `load_ppo_trainer`. Reason: combining a learned RM with a self-judge log-likelihood adds a third reward source that nobody has asked for; `reward_blend_alpha` already provides the "blend RM with verifier" capability.

This decision is intentionally restrictive. Lifting it requires a future
spec that names a concrete experiment that needs three reward sources.

## R2. Adapter signature: `score_batch(prompts, completions, gts) -> Tensor[B]` vs the existing per-sample `reward_fn(completion, gt) -> float`

**Question:** how does the new batched protocol coexist with the existing
per-sample protocol that `_RewardFnWrapper`, `make_reward_fn`, and the
diagnostic-fn dance at
[run_e2_7.py:248-253](../run_e2_7.py#L248-L253) all assume?

**Resolution:** RM-B implements **path 3** from
[reward_model_integration.md §"Adapter contract"](reward_model_integration.md#adapter-contract):
the trainer learns to call `score_batch` directly when
`self.reward_model_scorer is not None`, and falls back to the per-sample
`self.reward_fn(completion, gt)` loop otherwise. This avoids a stateful
adapter (no `_idx` counter) and keeps the per-sample path bit-identical
when capacity is `"none"`.

**Trade-off:** the diagnostic-fn dance at run_e2_7.py:248-253 swaps
`trainer.reward_fn` for the diagnostic fn during eval. Under capacity !=
"none", the diagnostic eval uses the swapped per-sample function (still
`gsm8k_reward`), so accuracy reporting stays correct. The training-reward
path uses `score_batch`. No conflict — these are different code paths.

## R3. Stateful `_RewardFnWrapper.set_questions` under DDP

**Question:** `_RewardFnWrapper` ([src/rewards.py:201-239](../../src/rewards.py#L201-L239))
is stateful — it has a `set_questions` method and an internal `_idx`
counter. The DDP migration already had to fix this (§7.6.2 hazard, now
LANDED at three call sites). Does `score_batch` cope with the same
sharding hazard?

**Resolution:** yes, trivially. `score_batch(prompts, completions, gts)`
takes lists in the order the trainer is processing them. Under DDP,
`generate_rollouts` ([ppo_trainer.py:267-389](../ppo_trainer.py#L267-L389))
shards `prompts` and `ground_truths` *before* the inner loop, so the
batched call site naturally receives the local shard. There is no
analog of `_idx`; the alignment is positional and per-call, not stateful.

## R4. Memory under DDP × RM at 8B

**Question:** does the spec's 8B × 3 mitigation table still hold given the
trainer growth?

**Resolution:** the trainer footprint has not grown. New code added since
the spec (P19 length-bucketed generation, the no_grad envelope, the
divisibility-pad in `evaluate`) does not allocate additional model copies.
The mitigation table at
[reward_model_integration.md §"Updated mitigation savings table at 8B"](reward_model_integration.md#updated-mitigation-savings-table-at-8b)
still reflects reality. `reward_model_reuse_reference=True` +
`gradient_checkpointing=True` + `optimizer_8bit=True` is still the
minimum stack for an 80 GB A100 to hold three 8B models.

The `accelerate_multi_gpu.yaml` `mixed_precision: bf16` × bf16-loaded
weights interaction (no fp32 master, §7.6.6 advisory) remains unresolved
and will need attention before the first 8B production run. Track in
the OOS-FOLLOWUP list.

## R5. Accuracy metric source under continuous reward

**Question:** where does the `accuracy` field in run-script output come
from when `reward_model_capacity != "none"`?

**Resolution:** *always* from `gsm8k_reward`. This is enforced in two
places:
- `train_step` ([ppo_trainer.py:933-935](../ppo_trainer.py#L933-L935))
  uses `compute_accuracy([r.reward for r in batch.rollouts])` today —
  under capacity != "none", `r.reward` is a continuous RM score. The
  accuracy metric becomes meaningless. RM-B's evaluator change forces
  the accuracy column to recompute via `gsm8k_reward` whenever
  `self.reward_model_scorer is not None`.
- `evaluate` ([ppo_trainer.py:954-1035](../ppo_trainer.py#L954-L1035))
  similarly recomputes accuracy from `gsm8k_reward` regardless of which
  scorer trained the policy.

The spec is explicit; RM-B's prompt repeats the requirement; D3 verifies it.

## R6. `wait_for_everyone()` around `score_batch` — needed?

**Question:** does the new batched RM call site need a barrier?

**Resolution:** **no**. `score_batch` is a no-grad forward pass on a
frozen model. It does not register parameters with DDP's reducer and
does not all-reduce gradients (none exist). It runs identically on every
rank because every rank holds the same gathered `RolloutBatch` post-§1.2
(see [ppo_trainer.py:365-369](../ppo_trainer.py#L365-L369)).

The only required collective is the existing `all_gather_object` at
[L368](../ppo_trainer.py#L368). After that, every rank sees the same
prompts/completions, runs `score_batch` on its local replica of the
frozen RM, and produces identical scores. No barrier needed before or
after the call.

The exception: if a future variant runs `score_batch` with the RM on
CPU offload (per the spec's mitigation #3), the on-device move /
off-device move forms an effective barrier-like pattern and may need
explicit synchronization. Not applicable to Phase 2.

## R7. Total rollout count regression risk

**Question:** the audit flagged that the saved-rollout-count dance at
[run_e2_7.py:280-282](../run_e2_7.py#L280-L282) and
[run_e2_8.py:158-162](../run_e2_8.py#L158-L162) interacts with the global
counter. Does the RM batched substitution preserve this?

**Resolution:** yes. RM-B does not touch the
`self.total_rollouts += global_B` increment at
[ppo_trainer.py:388](../ppo_trainer.py#L388). The dance — save
`saved_rollouts = trainer.total_rollouts`, run
`generate_rollouts(...)`, restore `trainer.total_rollouts =
saved_rollouts` — operates on the global count and remains symmetric.
RM-B's instructions explicitly forbid touching that counter. Re-verify
in code review.

## R8. Test-import order dependency

**Question:** the new `ppo_specs/reward_model.py` imports
`AutoModelForCausalLM`. The test file `test_reward_model.py`
monkeypatches `transformers.AutoModelForCausalLM.from_pretrained`. Will
the monkeypatch take effect if `reward_model.py` was already imported
elsewhere in the same Python process?

**Resolution:** yes — patching `from_pretrained` (a method on the class,
not the symbol exported from the module) survives any pre-existing
`from transformers import AutoModelForCausalLM` import in
`reward_model.py`, because `reward_model.AutoModelForCausalLM` is the
SAME class object as `transformers.AutoModelForCausalLM`. RM-TEST's
prompt explicitly instructs the agent to use
`monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_loader)`
rather than patching `ppo_specs.reward_model.AutoModelForCausalLM`,
which would be brittle.

## R9. CPU bf16 dtype interaction

**Question:** `accelerate_cpu.yaml` sets `mixed_precision: 'no'`. The
existing trainer dtype resolution at
[ppo_trainer.py:1064-1070](../ppo_trainer.py#L1064-L1070) picks bf16 when
`device.type == "cuda"` and fp32 on CPU. Does this hold for the RM load?

**Resolution:** RM-A's prompt says to respect `device.type` for
`reward_model_dtype="auto"` (bf16 on CUDA, fp32 on CPU). When
`reward_model_reuse_reference=True`, the RM inherits the reference
model's dtype (already correctly resolved by the existing reference
load at L1064-L1070), so this is automatic. When loading independently,
RM-A must mirror the same logic.

# Open design questions (escalate before Phase 2 starts)

These need a human decision; agents should NOT guess.

## Q1. Should DDP-CLEAN block DDP-F1?

The four ungated prints in `load_ppo_trainer` are noisy under multi-process
runs but functionally harmless. Recommendation: **defer to a follow-up
PR**, not a Phase 2 blocker. F1 smoke is robust to log clutter.

## Q2. Should we expose an explicit `score_batch_callable_adapter` for users who want to plug a learned RM into an existing trainer they don't own?

**Recommendation:** no, not in Phase 2. The two trainer call sites
(`generate_rollouts` and `evaluate`) are inside this repo; we own them.
External users can implement their own adapter if they need one. Keeping
the surface minimal helps with R2's "no stateful wrapper" decision.

## Q3. Does the SLURM `accelerate_multi_gpu.yaml` need `mixed_precision: 'no'` for the first 8B run?

The §7.6.6 advisory hazard. Recommendation: **set it to `'no'` in the
yaml** for 8B runs, since the model is loaded as bf16 directly. With
`mixed_precision: bf16`, Accelerate adds a no-op autocast — measurable
overhead, no correctness issue. Track as an explicit follow-up bead
**DDP-MP-8B** (out of scope for Phase 2 unless the 8B run lands inside
Phase 2).

## Q4. Should `DDP-BAR` add a unit test that scans the run scripts for `save_checkpoint` without a preceding `_wait()`?

**Recommendation:** yes, but lightweight — a string-level grep test in
`ppo_specs/tests/test_run_e2_7_ddp_signature.py` (which already exists)
that asserts the line containing `save_checkpoint(` has a `_wait()` or
`wait_for_everyone` call within the previous 10 lines. Catches future
regressions cheaply. Add as part of DDP-BAR's "Files to edit".
