# PPO DDP Migration — Sub-Agent Prompts

Companion to [ddp_cpu_gpu_migration.md](ddp_cpu_gpu_migration.md). These are
ready-to-dispatch prompts for each stage of the CPU-smoke-then-GPU migration.
Copy the prompt, paste into an Agent tool call, set the indicated
`subagent_type`. Each prompt is self-contained — assumes the agent has not
seen this conversation.

## Status (refreshed 2026-05-04)

Agents B, C, D have **already landed**. Agents A and E **have NOT**. F1 and F2
verification stages are still pending. The remaining work is the Phase 2
beads named **DDP-A** (= original Agent A's CPU yaml + setup_env.sh
`--cpu-only`), **DDP-E** (= original Agent E's `DEVICE_MODE` SLURM branch),
and **DDP-BAR** (audit/strengthen the barrier coverage given the run-script
drift).

`configs/accelerate_cpu.yaml` is already present — only the `setup_env.sh
--cpu-only` half of original Agent A still owes work. See
[integration_beads.md](integration_beads.md) for the refreshed bead
assignment.

**Dispatch order (original — still valid for any new clone):** A → (B, C, D,
E in parallel — different files, no shared state) → F1 → (smoke passes) → F2.

Agents B, C, D, E all depend on Agent A's `configs/accelerate_cpu.yaml` only
for verification; they can run in parallel against different files.

**Rollback:** if any of B/C/D/E fails or merges in a broken state, revert
ONLY that agent's commit; A is independent and stays. F1 (smoke) is the
gate that catches integration breakage between B/C/D/E — do not run F1
until all four have landed.

**Baseline invariant (added 2026-05-04):** the existing `reward_fn` /
`reward_mode` / `_RewardFnWrapper` / `make_reward_fn` /
`SelfJudgeRewardModel` plumbing stays. Any DDP change MUST preserve
bit-identical behavior in the single-process path. See
[ddp_cpu_gpu_migration.md §"Baseline invariant"](ddp_cpu_gpu_migration.md).

**Critical pitfalls** (each is documented in detail below; flagged here
for triage):
1. `B = len(local_prompts)` MUST be rebound after sharding in
   `generate_rollouts` (Agent B). Forgetting this is a silent IndexError
   on rank > 0. (LANDED at [ppo_trainer.py:293](../ppo_trainer.py#L293).)
2. `accelerator.wait_for_everyone()` MUST live in the run-script CALLER
   before every `save_checkpoint` (Agents C and D). Putting the barrier
   inside `save_checkpoint` deadlocks because non-rank-0 ranks return
   early. (LANDED via the `_wait()` helper.)
3. `import torch.distributed as dist` is a separate import from
   `from accelerate import Accelerator` (Agent B). `accelerator.gather_for_metrics`
   handles tensors only, not Python objects — `dist.all_gather_object`
   is required for the Rollout list. (LANDED at [ppo_trainer.py:43](../ppo_trainer.py#L43) and [L368](../ppo_trainer.py#L368).)
4. `reference_model` parameters must remain frozen
   (`requires_grad=False`); add a defensive assertion at trainer
   construction (Agent B). (LANDED at [ppo_trainer.py:199-200](../ppo_trainer.py#L199-L200).)

---

## Agent A — CPU accelerate config + `setup_env.sh --cpu-only`

Subagent type: `general-purpose`

**STATUS 2026-05-04:** Half of this work is done. `configs/accelerate_cpu.yaml`
already exists at the repo root (verify with `ls configs/accelerate_cpu.yaml`).
The remaining work is the `setup_env.sh --cpu-only` flag. The prompt below
is unchanged for fidelity but only step 2 is still actionable.

```
Task: Land the CPU-side infra for a CPU DDP smoke test of the PPO pipeline.
Repo root: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

Read first (context):
- ppo_specs/specs/ddp_cpu_gpu_migration.md §4 "New file: configs/accelerate_cpu.yaml"
- ppo_specs/specs/ddp_cpu_gpu_migration.md §6 "setup_env.sh --cpu-only"
- configs/accelerate_single_gpu.yaml, configs/accelerate_multi_gpu.yaml — the existing templates you are matching
- scripts/setup_env.sh — the script you are modifying

Do exactly two things:

1. Create configs/accelerate_cpu.yaml with the content specified in §4 of the
   spec:

     compute_environment: LOCAL_MACHINE
     debug: false
     distributed_type: MULTI_CPU
     downcast_bf16: 'no'
     machine_rank: 0
     main_training_function: main
     mixed_precision: 'no'
     num_machines: 1
     num_processes: 4
     rdzv_backend: static
     same_network: true
     use_cpu: true

   No additional keys. No comments inside the YAML (Accelerate rejects some
   comment placements).

2. Edit scripts/setup_env.sh per §6 of the spec:
   - Initialize CPU_ONLY=false near L23, alongside the other defaults.
   - Parser at L29-L39: add `--cpu-only) CPU_ONLY=true; shift ;;`.
   - Step 2 at L71-L95: if CPU_ONLY=true, set
       TORCH_INDEX="https://download.pytorch.org/whl/cpu"
     unconditionally (bypass the case statement), skip the CUDA verification
     block at L87-L95, and print only `torch.__version__`.
   - Step 4 (model pre-download): leave unchanged.
   - Footer: when CPU_ONLY=true, print a banner with the smoke command:
       accelerate launch --config_file configs/accelerate_cpu.yaml \
           ppo_specs/run_e2_7.py --local-test --no-mc

Invariants to preserve:
- The GPU path MUST remain the default (CPU_ONLY defaults to false).
- No other arguments to setup_env.sh may change behavior.
- No Python edits. No SLURM-script edits (Agent E owns slurm_*.sh).

Self-verification before finishing:
- `cat configs/accelerate_cpu.yaml` shows exactly the 12 keys above.
- `bash scripts/setup_env.sh --help 2>&1 | head -5` exits cleanly (or: `bash -n scripts/setup_env.sh` syntax-checks).
- `bash scripts/setup_env.sh --cpu-only --skip-models --env-name rlvr_cpu_test`
  on a dry-run basis: the TORCH_INDEX echoed is the CPU one. (If conda/activation
  noise is unavoidable, a shell trace with `bash -x` showing TORCH_INDEX=...cpu
  is sufficient evidence.)
- `bash scripts/setup_env.sh --skip-models` (no --cpu-only) still picks a cu*
  TORCH_INDEX.

Report: the two files touched, the TORCH_INDEX value observed under
--cpu-only, and any deviation from the spec with justification (should be
minimal).
```

---

## Agent B — Wire Accelerate into `PPOTrainer`

Subagent type: `general-purpose`

**STATUS 2026-05-04: LANDED.** This prompt is preserved verbatim for design
history; the work is done at the line numbers updated below. Re-run only if
re-implementing on a fresh worktree.

```
Task: Refactor PPOTrainer in ppo_specs/ppo_trainer.py to run under
HuggingFace Accelerate for single-proc, multi-proc CPU (gloo), and multi-GPU
DDP with no code-path divergence. Repo root:
c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

Read first (context):
- ppo_specs/specs/ddp_cpu_gpu_migration.md §1 "Trainer Accelerate integration"
- ppo_specs/specs/ddp_cpu_gpu_migration.md §8 "Risks and gotchas"
- ppo_specs/specs/distributed.md §3 "HuggingFace Accelerate Integration"
- ppo_specs/ppo_trainer.py — the file you are editing, all of it, but especially:
  - L154-L244 __init__
  - L267-L389 generate_rollouts
  - L473-L501 _batched_per_token_log_probs / _batched_sequence_log_probs helpers
  - L503-L624 _batched_critic_values, _extract_last_hidden, _eval_critic_on_prompts
  - L628-L854 ppo_update, _policy_log_probs, _critic_forward
  - L954-L1035 evaluate
  - L1040-L1197 load_ppo_trainer

Edits, in order:

1. Imports: add
     from accelerate import Accelerator
     import torch.distributed as dist

2. Module-level helper near the top, after imports:

     def _shard(items, rank: int, world_size: int):
         per_rank = len(items) // world_size  # divisibility asserted at startup
         return items[rank * per_rank : (rank + 1) * per_rank]

3. PPOTrainer.__init__ (L114-L163):
   - Change the signature: replace the `device: torch.device` parameter with
     `accelerator: Accelerator`.
   - At the top of the body, store `self.accelerator = accelerator` and
     `self.device = accelerator.device`.
   - Immediately after storing critic (L127), cache:
       self._critic_trainable = critic.is_trainable()
   - After the optimizer construction at L151-L159, call accelerator.prepare:
       self.model, self.policy_optimizer = accelerator.prepare(
           self.model, self.policy_optimizer)
       if self._critic_trainable:
           self.critic, self.critic_optimizer = accelerator.prepare(
               self.critic, self.critic_optimizer)
       else:
           self.critic = self.critic.to(self.device)
   - Reference model (L136-L140): replace the implicit device expectation with
     `reference_model.to(self.device)`. Leave it UNWRAPPED by Accelerate.
   - Defensive assertion (right after the `.to()` call): verify the reference
     model is actually frozen. If a future caller forgets `requires_grad_(False)`,
     wrapping into Accelerate later will silently start syncing zero gradients:
       if reference_model is not None:
           assert all(not p.requires_grad for p in reference_model.parameters()), \
               "reference_model must be frozen before passing to PPOTrainer"

4. generate_rollouts (L167-L244):
   - At entry, capture `global_B = len(prompts)`.
   - Get `rank = self.accelerator.process_index`, `ws = self.accelerator.num_processes`.
   - Shard:
       local_prompts = _shard(prompts, rank, ws)
       local_gts = _shard(ground_truths, rank, ws)
       B = len(local_prompts)   # CRITICAL: rebind so `for i in range(B)` iterates
                                # over the shard, not the global batch. Forgetting
                                # this is a silent IndexError on rank > 0.
     Rebind every reference to `prompts[i]` / `ground_truths[i]` in the body
     (including at the Rollout construction) to `local_prompts[i]` / `local_gts[i]`.
   - Replace the `self.model.generate(...)` call at L188-L195 with:
       unwrapped = self.accelerator.unwrap_model(self.model)
       out = unwrapped.generate(...)
   - After the per-sample rollout-building loop, BEFORE
     `_batched_sequence_log_probs` and `_batched_critic_values`, gather:
       if ws > 1:
           gathered = [None] * ws
           dist.all_gather_object(gathered, rollouts)
           rollouts = [r for shard in gathered for r in shard]
   - The log-probs and critic-value calls at L229-L241 now operate on the full
     gathered list — leave those unchanged.
   - Replace `self.total_rollouts += B` at L243 with `self.total_rollouts += global_B`.

5. Replace every call to `self.critic.is_trainable()` in the file with
   `self._critic_trainable`. Sites: L238, L281, L467, L518 (verify by grep).

6. ppo_update (L365-L471):
   - Replace `total_loss.backward()` at L457 with
     `self.accelerator.backward(total_loss)`.
   - Replace the two `torch.nn.utils.clip_grad_norm_(...)` calls at L464-L468 with:
       policy_grad_norm = self.accelerator.clip_grad_norm_(
           self.model.parameters(), max_norm=self.config.grad_clip_norm,
       )
       if self._critic_trainable:
           self.accelerator.clip_grad_norm_(
               self.critic.parameters(), max_norm=self.config.grad_clip_norm,
           )
   - Leave metric `.item()` calls at L480-L488 alone — identical batches on
     every rank means identical metrics; no reduction needed.

7. evaluate (L623-L665):
   - Shard eval_prompts/eval_gts immediately after the L632-L633 slicing:
       rank = self.accelerator.process_index
       ws = self.accelerator.num_processes
       eval_prompts = _shard(eval_prompts, rank, ws)
       eval_gts = _shard(eval_gts, rank, ws)
   - Replace `self.model.generate(...)` at L647-L653 with
     `self.accelerator.unwrap_model(self.model).generate(...)`.
   - Gather rewards at the end:
       if ws > 1:
           local = torch.tensor(rewards, device=self.device)
           all_rewards = self.accelerator.gather_for_metrics(local)
           return compute_accuracy(all_rewards.tolist())
       return compute_accuracy(rewards)

8. load_ppo_trainer (L670-L738):
   - Signature change: replace `device: torch.device` with
     `accelerator: Accelerator`. Add
     `from accelerate import Accelerator` to the imports at top of file.
   - Replace `device.type` with `accelerator.device.type` at L680.
   - Gate the prints at L686, L703, L706-L707, L720-L721 on
     `accelerator.is_main_process`.
   - Drop `.to(device)` on the model (L696) and critic (L711). Accelerate's
     prepare() inside __init__ handles placement.
   - Keep `.to(accelerator.device)` on reference_model (L725).
   - At the final return, pass `accelerator=accelerator` to PPOTrainer(...)
     instead of `device=device`.

Invariants to preserve:
- Every generation site (rollout + eval) uses `accelerator.unwrap_model(...)`.
- reference_model stays unwrapped; `.to(accelerator.device)` explicitly.
- `self._critic_trainable` is read instead of `self.critic.is_trainable()`
  everywhere after construction.
- `total_rollouts` increments by GLOBAL batch size, not local shard.
- No dangling `.to(device)` on the policy or critic — Accelerate owns placement.

Do NOT edit:
- ppo_specs/run_e2_7.py, ppo_specs/run_e2_8.py (Agent C)
- ppo_specs/checkpoint.py (Agent D)
- scripts/* (Agent E)

Self-verification:
- `python -c "from ppo_specs.ppo_trainer import load_ppo_trainer, PPOTrainer; print('ok')"`
  succeeds.
- `grep -n 'self.critic.is_trainable()' ppo_specs/ppo_trainer.py` returns
  ZERO hits (all replaced by `self._critic_trainable`).
- `grep -n '\.to(device)' ppo_specs/ppo_trainer.py` returns zero hits
  inside load_ppo_trainer. The only legitimate `.to(...)` remaining should
  be on reference_model and inside _batched_per_token_log_probs (which uses
  self.device, a proxy for accelerator.device).
- `grep -n 'total_loss.backward' ppo_specs/ppo_trainer.py` returns zero
  hits; replaced by `self.accelerator.backward(total_loss)`.
- `grep -n 'torch.nn.utils.clip_grad_norm_' ppo_specs/ppo_trainer.py`
  returns zero hits.

Report: line ranges touched in each of the eight sections, the grep
confirmations above, and any signature-change fallout you had to propagate
beyond the listed sites.
```

---

## Agent C — Run scripts: Accelerator init, rank-0 gating, barriers

Subagent type: `general-purpose`

**STATUS 2026-05-04: LANDED.** Run scripts already use `_print`/`_wait`/`_is_main`
helpers (functionally equivalent to direct `accelerator.is_main_process` calls
the spec describes). The remaining gap is the **DDP-BAR** bead — verify that
every checkpoint save site has a `_wait()` predecessor, including any future
checkpoint sites added to `run_e2_8.py`.

```
Task: Update ppo_specs/run_e2_7.py and ppo_specs/run_e2_8.py to run under
Accelerate. Repo root:
c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.
Assumes Agent B's PPOTrainer refactor has landed (signature change to
`accelerator: Accelerator`).

Read first:
- ppo_specs/specs/ddp_cpu_gpu_migration.md §2 "Run-script integration"
- ppo_specs/specs/ddp_cpu_gpu_migration.md §3 "Config + divisibility"
- ppo_specs/run_e2_7.py, esp. L55-L367 (whole main function)
- ppo_specs/run_e2_8.py, esp. L215-L357 (whole sweep function) and L72-L210
  (run_one_capacity helper)

Changes to ppo_specs/run_e2_7.py, inside run_e2_7() (starts L55):

1. Replace L56-L63 (device selection + manual seeding) with:

     from accelerate import Accelerator
     from accelerate.utils import set_seed

     accelerator = Accelerator()
     device = accelerator.device

     assert config.batch_size % accelerator.num_processes == 0, (
         f"batch_size ({config.batch_size}) must be divisible by "
         f"num_processes ({accelerator.num_processes})"
     )

     set_seed(config.seed)

   (`device` is kept as a local name only so you do not have to rewrite the
   rest of the function; you will delete the redundant uses next.)

2. Pass `accelerator` to `load_ppo_trainer` at L76:
     trainer = load_ppo_trainer(config, accelerator)

3. Wrap every print / logger.save / save_checkpoint call in the rank-0 gate.
   Exact sites (use the spec's §2.1 table):

     L64, L67, L93, L118, L128, L137, L156-L164, L225-L229, L237, L250-L251
       → print — wrap with `if accelerator.is_main_process:`
     L223 logger.log_step — gate
     L224, L239, L246 logger.save — gate
     L233, L238, L244 save_checkpoint — gate AND precede with
       accelerator.wait_for_everyone()

   Idiomatic pattern (do this once near the top and reuse):

     def print0(*args, **kwargs):
         if accelerator.is_main_process:
             print(*args, **kwargs)

   Then replace raw `print(...)` with `print0(...)` at the listed sites.
   This keeps the diff readable.

4. The exit_handler path at L236-L240: gate both the print AND the
   save_checkpoint/logger.save; the wait_for_everyone() should come
   BEFORE the if-main-process block so all ranks reach the barrier.

Changes to ppo_specs/run_e2_8.py:

5. Inside run_e2_8() (starts L178):
   - L179-L186: replace device selection and manual seeding with the same
     Accelerator()/set_seed pattern as run_e2_7 step 1.
   - Pass `accelerator` instead of `device` to any load_ppo_trainer call
     (L200 in the tmp_trainer build).
   - L236-L241: replace the per-capacity manual re-seeding with `set_seed(config.seed)`.
   - Rank-0 gates per the spec's §2.2 table:
       L187, L190, L213, L230, L254-L257, L259-L269 — gate on is_main_process
       L159 logger.save (inside run_one_capacity) — gate
       L165 save_checkpoint (if present inside run_one_capacity) — gate AND
         precede with wait_for_everyone()
   - wait_for_everyone() before the summary-file write at L254-L257.
   - Gc/empty_cache dance at L222-L228: keep unchanged, but note that
     torch.cuda.empty_cache() is already guarded by is_available(), so it's
     a no-op on CPU — no change needed.

6. Inside run_one_capacity() (spans L70-L170 approximately):
   - Per-step prints at L80-L82: gate.
   - L153-L157 final accuracy print: gate.
   - Any other prints that appear between L70-L170: gate them all (treat
     this helper as "runs on every rank but only rank 0 prints").
   - The inner load_ppo_trainer call inside run_one_capacity needs the
     accelerator too — thread `accelerator` through as a parameter, OR
     import it via the run_e2_8 scope (pick one and be consistent).

7. ppo_specs/config.py: one-line comment at the batch_size field
   (approx L60):

     batch_size: int = 16  # GLOBAL batch size; must be divisible by Accelerator.num_processes

   No other config changes.

Invariants to preserve:
- Every print, log_step, save, and save_checkpoint is gated on rank 0.
- Every save_checkpoint is preceded by accelerator.wait_for_everyone().
- `set_seed(config.seed)` replaces all manual RNG seeding (torch, numpy,
  python random, transformers_set_seed, cuda.manual_seed_all).
- `config.batch_size % num_processes == 0` assertion fires at startup on
  every rank.
- The saved_rollouts dance at run_e2_7.py:195-197 and run_e2_8.py:122-127
  is UNCHANGED — with Agent B's fix, `saved_rollouts` captures the GLOBAL
  count and the restoration is symmetric.

Do NOT edit:
- ppo_specs/ppo_trainer.py (Agent B)
- ppo_specs/checkpoint.py (Agent D)
- scripts/*, configs/* (Agents A, E)

Self-verification:
- `python -c "import ppo_specs.run_e2_7; import ppo_specs.run_e2_8; print('ok')"`
  succeeds (no import errors from the signature change).
- `grep -n 'torch.manual_seed\|torch.cuda.manual_seed_all\|np.random.seed' ppo_specs/run_e2_7.py ppo_specs/run_e2_8.py`
  returns zero hits (all replaced by set_seed).
- `grep -cn 'accelerator.is_main_process' ppo_specs/run_e2_7.py` returns
  a count >= 12 (approx; one per listed gate site).
- `grep -n 'accelerator.wait_for_everyone' ppo_specs/run_e2_7.py ppo_specs/run_e2_8.py`
  returns >= 4 hits (3 save_checkpoint + 1 summary write).
- Single-proc smoke still works:
    PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc
  should complete 5 steps and print final_acc on stdout (rank 0 is the only
  rank so every print fires).

Report: the number of rank-0 gates added in each file, the set_seed
conversion count, any helper (print0 etc.) you introduced, and any line
drift vs. the spec's tables (spec was built on a snapshot — minor 1-2-line
drift is expected).
```

---

## Agent D — Checkpoint unwrap + rank-0 write gate

Subagent type: `general-purpose`

**STATUS 2026-05-04: LANDED.** `save_checkpoint` now accepts an optional
`accelerator` kwarg, unwraps via `accelerator.unwrap_model` when provided,
and documents the caller-gated barrier contract in its docstring. The
rank-0 write gate is delegated to the caller per the design (see the
docstring at [checkpoint.py:50-63](../checkpoint.py#L50-L63)).

```
Task: Update ppo_specs/checkpoint.py so that save_checkpoint is safe under
multi-process Accelerate runs. Repo root:
c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

Read first:
- ppo_specs/specs/ddp_cpu_gpu_migration.md §1.1, §1.2 (PPOTrainer changes),
  §2 (run-script gating), §8 (gotchas, esp. _config_hash stability)
- ppo_specs/checkpoint.py — the file you are editing
- ppo_specs/specs/checkpointing.md (deeper background)

Edits to ppo_specs/checkpoint.py:

1. save_checkpoint (L28-L130):
   - At function entry, fetch the accelerator from the trainer:
       accelerator = getattr(trainer, "accelerator", None)
     This makes the function safe for BOTH pre-Accelerate single-proc code
     paths (accelerator=None → behave as before) AND the new multi-proc
     path.
   - Unwrap model and critic before state_dict():
       model_to_save = (
           accelerator.unwrap_model(trainer.model)
           if accelerator is not None else trainer.model
       )
       critic_to_save = (
           accelerator.unwrap_model(trainer.critic)
           if accelerator is not None else trainer.critic
       )
     Replace trainer.model.save_pretrained at L47 with
     model_to_save.save_pretrained(...).
     Replace trainer.critic.state_dict() at L51 with critic_to_save.state_dict().

   - Gate the actual filesystem writes on rank 0:
       if accelerator is not None and not accelerator.is_main_process:
           return str(ckpt_path)  # other ranks return the path silently
       # ... existing save logic ...

     The barrier is the run-script's responsibility (§2 of the spec adds
     accelerator.wait_for_everyone() before every save_checkpoint call),
     so this function does NOT need to call wait_for_everyone itself.

2. _config_hash helper (definition at ppo_specs/checkpoint.py:210-230, called
   at checkpoint.py:97 and validation at L147):
   - Ensure the hash is computed on the GLOBAL config (in particular
     batch_size is the global value, which is already what PPOConfig
     stores — no change needed, just verify).
   - Add a brief comment explaining the invariant:
       # batch_size here is the GLOBAL size, so the hash is stable across
       # world sizes — a 4-proc resume from a 1-proc checkpoint has the
       # same hash.

3. load_checkpoint (L133-onward):
   - Must be callable on all ranks (every rank needs to load the model
     weights). Do NOT gate load on rank 0 — that would leave other ranks
     with uninitialized weights and no broadcast.
   - After loading state dicts into trainer.model / trainer.critic, if
     trainer.accelerator is not None, the prepared-but-unwrapped reassembly
     is handled by Accelerate automatically; no explicit re-prepare needed.
   - RNG state restoration: restore on every rank. Per-rank RNG streams
     diverged during the run; re-restoring the same state on every rank
     produces correct resume behavior for the FIRST step (after which
     streams diverge again).

4. _rotate_checkpoints (the rotation helper): must also only run on rank 0
   when accelerator is present. Add the gate at the top of the function, or
   have save_checkpoint only call it on rank 0.

Invariants to preserve:
- Checkpoint content schema is unchanged (model/critic/optimizers/RNG/logger/config).
- Atomic-rename semantics at L87-L90 still apply.
- A checkpoint written by a 4-proc run is loadable by a 1-proc run and
  vice-versa (`_config_hash` stable).
- Non-rank-0 ranks MUST reach the same barrier as rank 0 — do not `return
  None` before `accelerator.wait_for_everyone()` in the caller. (Caller
  responsibility per §2; reflect in a comment here.)

Do NOT edit:
- ppo_specs/ppo_trainer.py (Agent B)
- ppo_specs/run_e2_7.py, run_e2_8.py (Agent C)
- configs/*, scripts/* (Agents A, E)

Self-verification:
- `python -c "from ppo_specs.checkpoint import save_checkpoint, load_checkpoint, _config_hash; print('ok')"`
  succeeds.
- Single-proc smoke:
    PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc --checkpoint-every 2
  writes a checkpoint, and a subsequent run with `--resume-from auto`
  resumes at step 3+ and completes.
- `grep -n 'accelerator.unwrap_model' ppo_specs/checkpoint.py` shows at
  least 2 hits (model and critic).
- `grep -n 'is_main_process' ppo_specs/checkpoint.py` shows at least 1
  hit (the write gate).

Report: line ranges changed, the 1-proc resume test result, and any place
where the unwrap is NOT needed (e.g., reference model is not saved; verify
that's still the case).
```

---

## Agent E — SLURM parameterization (`DEVICE_MODE` branch)

Subagent type: `general-purpose`

**STATUS 2026-05-04: NOT DONE.** `configs/accelerate_cpu.yaml` is in place
but neither `slurm_e2_7.sh` nor `slurm_e2_8.sh` branches on `DEVICE_MODE`.
This is the **DDP-E** bead in [integration_beads.md](integration_beads.md).

```
Task: Add a DEVICE_MODE branch to scripts/slurm_e2_7.sh and
scripts/slurm_e2_8.sh so the same scripts run CPU smoke jobs and GPU
production jobs. Repo root:
c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

Read first:
- ppo_specs/specs/ddp_cpu_gpu_migration.md §5 "SLURM parameterization"
- scripts/slurm_e2_7.sh (158 lines)
- scripts/slurm_e2_8.sh (178 lines)
- configs/accelerate_cpu.yaml (already present at the repo root)

Edits to BOTH scripts/slurm_e2_7.sh and scripts/slurm_e2_8.sh:

1. Near the top of the configurable-parameters block (slurm_e2_7.sh L40-L50;
   slurm_e2_8.sh L40-L51), add:

     DEVICE_MODE="${DEVICE_MODE:-gpu}"        # gpu | cpu
     NUM_PROCESSES="${NUM_PROCESSES:-4}"      # accelerate --num_processes

2. In the derived-settings block (after the existing SLURM_MODE branch),
   add:

     if [ "$DEVICE_MODE" = "cpu" ]; then
         ACCEL_CONFIG="configs/accelerate_cpu.yaml"
     else
         ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_multi_gpu.yaml}"
     fi

3. Gate the `module load cuda/12.1` call at slurm_e2_7.sh:72 (and
   slurm_e2_8.sh:75) behind `DEVICE_MODE != cpu`:

     if [ "$DEVICE_MODE" != "cpu" ]; then
         module load cuda/12.1 2>/dev/null || true
     fi

4. Gate the NCCL block at slurm_e2_7.sh:86-93 (and slurm_e2_8.sh:88-93) behind
   `DEVICE_MODE != cpu`:

     if [ "$DEVICE_MODE" != "cpu" ]; then
         export NCCL_DEBUG=WARN
         export NCCL_IB_DISABLE=0
         export NCCL_NET_GDR_LEVEL=5
         export NCCL_SOCKET_IFNAME=eth0
         export NCCL_P2P_LEVEL=NVL
         export NCCL_TIMEOUT=1800
     fi

5. Runtime sanity warning, just after the DEVICE_MODE branch in step 2:

     if [ "$DEVICE_MODE" = "cpu" ] && [[ "${SLURM_JOB_GRES:-}" == *gpu* ]]; then
         echo "[WARN] DEVICE_MODE=cpu but SLURM_JOB_GRES=${SLURM_JOB_GRES} — wasting a GPU."
     fi

6. Unify the command-build block. Replace the existing single-GPU direct-
   python path at slurm_e2_7.sh:137-142 (currently `python ppo_specs/run_e2_7.py`)
   and the multigpu path at slurm_e2_7.sh:144-153 with ONE
   accelerate launch. In slurm_e2_8.sh, replace the per-capacity branch at
   L152-L153 (`python ppo_specs/run_e2_8.py`) similarly:

     echo "[RUN] accelerate launch --config_file $ACCEL_CONFIG --num_processes $NUM_PROCESSES ppo_specs/run_e2_7.py ${ARGS}"
     accelerate launch \
         --config_file "$ACCEL_CONFIG" \
         --num_processes "$NUM_PROCESSES" \
         ppo_specs/run_e2_7.py ${ARGS} &
     CHILD_PID=$!
     wait $CHILD_PID

   Keep the SLURM_MODE array-selection logic (seeds from ${SEEDS[@]}) in
   place — it only sets ARGS, not the launcher. Remove any old `python
   ppo_specs/run_e2_*.py ${ARGS}` invocations.

7. Do the same transformation in slurm_e2_8.sh — the file structure is
   parallel; replicate the edits 1:1.

Invariants to preserve:
- Default (no env vars set) behavior is unchanged: DEVICE_MODE=gpu,
  ACCEL_CONFIG=multi_gpu, NCCL exports fire, CUDA module loads.
- The SLURM_MODE=single / multigpu / array branches inside slurm_e2_7.sh
  still work — they control ARGS and seeding, not the launcher.
- The preemption handler at slurm_e2_7.sh:98-108 is unchanged (known bug,
  tracked separately).
- The #SBATCH directives at the top are UNCHANGED. Resource overrides
  (--gres, --partition) go at `sbatch` submission time.

Do NOT edit:
- configs/* (Agent A owns configs/accelerate_cpu.yaml)
- Any Python files (Agents B, C, D)
- Preemption handler (separate follow-up)

Self-verification:
- `bash -n scripts/slurm_e2_7.sh` and `bash -n scripts/slurm_e2_8.sh`
  both syntax-check cleanly (no submission).
- Dry-run the env-var branching logic:
    DEVICE_MODE=cpu bash -x scripts/slurm_e2_7.sh 2>&1 | head -60
  shows ACCEL_CONFIG=configs/accelerate_cpu.yaml and no `module load cuda`
  line. (Expect it to fail later when trying to find accelerate or conda;
  the env-var branching evidence is sufficient.)
    DEVICE_MODE=gpu bash -x scripts/slurm_e2_7.sh 2>&1 | head -60
  shows ACCEL_CONFIG=configs/accelerate_multi_gpu.yaml (or existing default)
  and `module load cuda/12.1` attempted.
- `grep -n 'module load cuda' scripts/slurm_e2_7.sh` shows the call wrapped
  in an `if` block.

Report: the exact `echo "[RUN] ..."` line emitted by each script under
DEVICE_MODE=cpu, any deviation from the spec with justification, and any
SLURM_MODE × DEVICE_MODE interactions you had to reason about beyond what
the spec specifies.
```

---

## Agent F1 — CPU smoke verification (no code changes)

Subagent type: `general-purpose`

```
Task: Verify the CPU DDP smoke works end-to-end across the five recipes in
ppo_specs/specs/ddp_cpu_gpu_migration.md §7. Repo root:
c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. This is
a verification task; make NO code changes.

Read first:
- ppo_specs/specs/ddp_cpu_gpu_migration.md §7 "Smoke-test recipe"
- ppo_specs/specs/ddp_cpu_gpu_migration.md §9 "Verification checklist"
- ppo_specs/config.py — local_test_config (batch_size=4, n_steps=5)

Runs, in order. Capture final_acc, wall time, and stdout rank-prefix
characteristics (only rank 0 prints?) for each.

1. Single-process CPU baseline:

     PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc

   Record: final_acc, wall time, any errors. Note whether total_rollouts
   in results/*.json equals batch_size × n_steps (4 × 5 = 20).

2. 4-process CPU DDP, run_e2_7:

     PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_cpu.yaml \
         ppo_specs/run_e2_7.py --local-test --no-mc

   Record: final_acc, wall time, whether only rank 0 printed (scan stdout
   for duplicated "[E2.7]" lines — should see each exactly once). Confirm
   results/*.json was written exactly once (no rank-N shadow files).

   Pass criteria per §7.2:
   - No hang on .generate() (would manifest as process stuck for > 10x run 1
     wall time).
   - No AttributeError on critic.is_trainable.
   - final_acc within numerical noise of run 1 (exact match not required).

3. 4-process CPU DDP, run_e2_8:

     PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_cpu.yaml \
         ppo_specs/run_e2_8.py --local-test

   Record: per-capacity final_accs and sweep total wall time. Confirm
   e2_8_sweep_summary.json written once, well-formed JSON.

4. CPU cluster sbatch dry-run (do NOT submit to real SLURM; instead run
   the submission-simulating dry-run):

     DEVICE_MODE=cpu NUM_PROCESSES=4 LOCAL_TEST=true \
         bash -x scripts/slurm_e2_7.sh 2>&1 | head -100

   (This will fail once it tries to call `scontrol` or `module load`, but
   the early env-var branching should print ACCEL_CONFIG=configs/accelerate_cpu.yaml
   and show the accelerate launch command constructed correctly.)

   Record: the exact `[RUN]` command echoed, presence or absence of
   `module load cuda` (should be absent under DEVICE_MODE=cpu).

5. GPU cluster sbatch (SKIP if no GPU cluster available in the verification
   environment; report "N/A" if so). If a GPU is available:

     PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
         ppo_specs/run_e2_7.py --local-test --no-mc

   Record: whether the run completes without any code change vs. run 2.
   This is the key payoff of the CPU smoke — if run 5 breaks with a bug not
   seen in run 2, report back as a smoke-coverage gap.

Also verify no unexpected side effects:
- `git status` after all runs: only `results/`, `logs/`, `checkpoints/`,
  `.model_cache/` (untracked) changes. No Python file modifications.
- No new files outside results/ and logs/.
- No Python warnings about "requires_grad" on frozen weights.
- No NCCL warnings or timeouts.

Produce a results table:

| Run | Procs | final_acc | Wall time | Rank-0-only stdout? | Notes |
|-----|-------|-----------|-----------|---------------------|-------|
|  1  |   1   |           |           |      (N/A)          |       |
|  2  |   4   |           |           |                     |       |
|  3  |   4   |           |           |                     |       |
|  4  |   4   |    N/A    | dry-run   |      (N/A)          |       |
|  5  |  (GPU) |          |           |                     |       |

Report:
- The table above, fully filled.
- Any run that hung, OOMed, or errored — include stdout tail for each.
- final_acc delta between run 1 and run 2 (numerical noise expected, but
  flag >0.1 absolute difference as suspicious).
- Any unexpected files in git status.
- Recommendations for spec §9 checklist items NOT yet verified.

Do NOT mark this task complete if any run fails.
```

---

## Agent F2 — Final code review (superpowers:code-reviewer)

Subagent type: `superpowers:code-reviewer`

```
Task: Review the complete CPU-to-GPU DDP migration of the PPO pipeline.
Repo root: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

The plan being executed is ppo_specs/specs/ddp_cpu_gpu_migration.md. All
implementation has landed across:

- NEW configs/accelerate_cpu.yaml
- MODIFIED scripts/setup_env.sh (`--cpu-only` flag)
- MODIFIED ppo_specs/ppo_trainer.py (Accelerate integration)
- MODIFIED ppo_specs/run_e2_7.py, ppo_specs/run_e2_8.py (rank-0 gates,
  set_seed, divisibility assert)
- MODIFIED ppo_specs/config.py (one-line comment)
- MODIFIED ppo_specs/checkpoint.py (unwrap + rank-0 write gate)
- MODIFIED scripts/slurm_e2_7.sh, scripts/slurm_e2_8.sh (DEVICE_MODE branch)

Review against the spec's §9 "Verification checklist" and §1 "Trainer
Accelerate integration", §2 "Run-script integration". Focus areas:

1. Accelerate integration correctness:
   - PPOTrainer.__init__ signature is `accelerator: Accelerator`, not a
     `device` parameter.
   - `accelerator.prepare(model, optimizer)` is called exactly once per
     trainable module; the frozen reference_model is NOT prepared and uses
     `.to(accelerator.device)` instead.
   - `self._critic_trainable` is cached before prepare and used everywhere
     downstream — `critic.is_trainable()` is never called after prepare.
   - `accelerator.backward(total_loss)` replaces `total_loss.backward()`.
   - `accelerator.clip_grad_norm_` replaces `torch.nn.utils.clip_grad_norm_`
     for BOTH policy and critic.

2. Unwrap at every generation site:
   - generate_rollouts uses `accelerator.unwrap_model(self.model).generate(...)`.
   - evaluate uses `accelerator.unwrap_model(...).generate(...)`.
   - No bare `self.model.generate(...)` or `self.model(input_ids=...)` for
     generation remains in ppo_trainer.py.
   - The no-grad forward pass inside `_batched_per_token_log_probs` may use
     `self.model` directly (it is a training-mode forward, not generate)
     — confirm this is safe under DDP: it IS safe because the forward is
     synchronous across ranks (same shape gathered batch).

3. Rank-0 gating completeness:
   - Every `print` in run_e2_7.py and run_e2_8.py inside main functions
     is gated. Use the spec's §2.1 and §2.2 tables as a checklist.
   - Every `logger.save()` is gated.
   - Every `save_checkpoint(...)` is preceded by
     `accelerator.wait_for_everyone()` AND gated.
   - checkpoint.py gates the write inside save_checkpoint itself (defense
     in depth).

4. Determinism invariants:
   - `accelerate.utils.set_seed(config.seed)` replaces all manual seeding
     (torch, numpy, random, transformers_set_seed, cuda.manual_seed_all).
   - No lingering `torch.manual_seed` or `np.random.seed` in run_e2_7.py
     or run_e2_8.py.
   - The sharded `Rollout` lists are reassembled in rank order via
     `all_gather_object` — verify the flattening preserves order
     `[r for shard in gathered for r in shard]` (not set-like ordering).

5. Shard divisibility:
   - `assert config.batch_size % accelerator.num_processes == 0` fires at
     startup in BOTH run scripts.
   - The `_shard` helper uses integer division; any residual handling
     (items % ws > 0) is prevented by the assert, not silently dropped.

6. No dangling `.to(device)` that ignores Accelerator's placement:
   - load_ppo_trainer does NOT call `.to(device)` on `model` or `critic`.
   - The only `.to(accelerator.device)` calls on modules are on
     `reference_model` (frozen, unwrapped, by design).
   - Tensor-level `.to(self.device)` inside the class body is fine
     (self.device is proxy for accelerator.device).

7. Checkpoint hash stability:
   - `_config_hash` hashes the global config (including batch_size as
     written, which is the global size). Resuming a 4-proc run from a
     1-proc checkpoint should succeed.

8. SLURM DEVICE_MODE branching:
   - `module load cuda/12.1` is gated on `DEVICE_MODE != cpu`.
   - NCCL exports are gated on `DEVICE_MODE != cpu`.
   - The single unified `accelerate launch` command correctly resolves
     ACCEL_CONFIG and NUM_PROCESSES.
   - Default `DEVICE_MODE=gpu` preserves pre-change behavior exactly.

9. `total_rollouts` accounting:
   - Incremented by global pre-shard count in generate_rollouts.
   - saved_rollouts dance at run_e2_7.py:195-197 and run_e2_8.py:122-127
     still works (because `saved_rollouts` captures the GLOBAL count after
     the fix, not a local count).

10. Spec adherence:
    - Any silent deviation from ddp_cpu_gpu_migration.md that was not
      justified in an implementer's report is flagged.
    - Out-of-scope items (§10) were NOT accidentally implemented
      (no DeepSpeed changes, no FSDP, no multi-node, no W&B, no
      SIGUSR1-handler changes).

Deliver a prioritized punch list:
- Blockers (must fix before merge). Example classes: any missing unwrap
  at a generation site; any ungated filesystem write; any `critic.is_trainable()`
  that survives past prepare; any `total_loss.backward()` that survives.
- Non-blocking issues (cleanup, edge cases, smoke coverage gaps).
- Nits (style, docstring drift, comments).

For each item: file:line, a one-sentence description, and the recommended
fix. Under 900 words total.
```

---

## Notes for the dispatcher

- **A must land before E** (E references the file A creates,
  `configs/accelerate_cpu.yaml`, by path). Strictly speaking E only needs
  the filename, but it's cleaner to serialize.
- **B, C, D** can run in parallel against separate files (ppo_trainer.py,
  run_e2_*.py + config.py, checkpoint.py). They share no source files.
- **C depends on B's signature change** to `load_ppo_trainer(config,
  accelerator)`. If B has not finished when C starts, C will hit an import
  error on the test-import step. Either serialize B → C, or dispatch both
  in worktrees and reconcile.
- **F1 runs after A, B, C, D, E all merge.** F1 verifies the merged system;
  dispatching F1 against partial merges yields confusing failures.
- **Do not dispatch F2 (code review) until F1 passes.** Reviewing code that
  hasn't been smoke-tested wastes cycles on issues the smoke would have
  caught.
- **Worktrees:** consider dispatching B, C, D with `isolation: "worktree"`
  so each has a clean copy of the repo. A and E touch separate files and
  can share a worktree if convenient.
- **Reporting discipline:** each agent starts from scratch with no context
  other than the prompt and the repo state. Re-paste the prompt verbatim;
  do not try to "continue from previous agent's work" in-band — use
  explicit artifacts (committed files, test output) as the handoff medium.
- **Track context:** agents' summaries arrive as tool results. Verify each
  by reading the changed files and running the self-verification commands
  yourself before declaring the stage complete.
