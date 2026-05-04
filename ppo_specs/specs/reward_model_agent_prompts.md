# Reward Model Integration — Sub-Agent Prompts

Companion to [reward_model_integration.md](reward_model_integration.md). These are ready-to-dispatch prompts for each stage of the RM work. Copy the prompt, paste into an Agent tool call, set the indicated `subagent_type`. Each prompt is self-contained — assumes the agent has not seen this conversation.

**Dispatch order**: A → B → C → D1 → (smoke passes) → (D2 ‖ D3) → D4. A and B can run in parallel if given clean worktrees; otherwise serialize since both edit `ppo_specs/config.py`. D2 (memory pre-flight) and D3 (code review) are independent and can run in parallel after D1 passes.

**Read this first**: [reward_model_integration.md §"Current implementation status"](reward_model_integration.md). The repo already contains a partial reward layer (`SelfJudgeRewardModel`, `_RewardFnWrapper`, `make_reward_fn`, plus `reward_mode`/`self_judge_*`/`reference_kl_coeff` config). Agents working on this spec build the TIER-BASED LEARNED RM on top of those primitives — they do NOT replace them. The "none" capacity wraps `gsm8k_reward`; the existing `reward_mode="self_judge"` path stays untouched.

**Critical pitfalls**:
1. `_config_hash` in [checkpoint.py:210-230](../checkpoint.py#L210-L230) MUST be updated with the new RM fields. THREE of five are already wired via `getattr` defaults (`reward_model_capacity`, `reward_blend_alpha`, `reward_score_activation` — see [checkpoint.py:223-225](../checkpoint.py#L223-L225)); the TWO MISSING fields are `reward_model_name` and `reward_model_reuse_reference`. Agent A adds those plus the corresponding `PPOConfig` fields. D3 verifies it.
2. The reported `accuracy` metric MUST always come from `gsm8k_reward`, not from the RM. Training reward (`mean_reward`) and accuracy are different signals when `capacity != "none"`. Agent B's wiring must preserve this.
3. `reward_score_activation` defaults to `"sigmoid"` so RM output is in `[0, 1]` and blends safely with `gsm8k_reward ∈ {0, 1}`. Picking `"none"` with `reward_blend_alpha < 1.0` is unsafe unless the RM's output range is measured.
4. **Baseline invariant** (locked 2026-05-04): `reward_mode` and the existing `_RewardFnWrapper` / `make_reward_fn` / `SelfJudgeRewardModel` plumbing stay. The new `RewardModelScorer` plugs into the same `reward_fn` slot. When `reward_model_capacity == "none"`, every code path is bit-identical to today. See [reward_model_integration.md §Scope](reward_model_integration.md#scope) for the full text.

---

## Agent A — Implement the reward model module and factory

Subagent type: `general-purpose`

```
Task: Implement the learned reward model module for a PPO pipeline. Repo root: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

Read first (context):
- ppo_specs/specs/reward_model_integration.md — the design spec you are implementing (especially §Scope > "Baseline invariant" and §"Interaction with reward_mode")
- ppo_specs/critic.py — the factory/capacity-tier pattern to mirror
- ppo_specs/ppo_trainer.py L1040-L1197 (load_ppo_trainer) — the model-loading pattern to mirror
- ppo_specs/ppo_trainer.py L514-L571 (_extract_last_hidden) — the batched-LM-forward pattern
- shared/per_token_loss.py — reusable batched LM helpers
- src/rewards.py L84-L105 — the gsm8k_reward function you are wrapping in the "none" tier
- src/rewards.py L201-L239 — the existing _RewardFnWrapper that the new layer must NOT replace (Baseline invariant)
- src/rewards.py L242-L285 — make_reward_fn factory; this stays as-is, you compose on top of it

Create a new file ppo_specs/reward_model.py containing:

1. A RewardModelScorer(nn.Module) class with method
   score_batch(prompts, completions, ground_truths=None) -> torch.Tensor of shape [B].

2. Three capacity tiers implemented as subclasses:
   - NoneReward: wraps src.rewards.gsm8k_reward, iterates over the batch. Raises ValueError if ground_truths is None.
   - LearnedRMScorer: loads AutoModelForCausalLM + a linear value head, uses left-padded batched tokenization of (prompt + completion), runs a no-grad forward, extracts the last-real-token hidden state via the attention mask (mirror _extract_last_hidden), passes through the value head, returns [B] float. Tokenizer mismatch tolerated (it re-tokenizes from strings). All parameters have requires_grad_(False).
   - Support a reuse_base_model kwarg: if provided, skip from_pretrained and wrap the given model as base.

3. A build_reward_model(config, device, *, base_model=None) factory mirroring build_critic. Dispatches on config.reward_model_capacity. When capacity != "none":
   - Picks dtype by respecting config.reward_model_dtype ("auto" -> bf16 on CUDA, fp32 on CPU).
   - If config.reward_model_reuse_reference is True and base_model is not None, wraps base_model; else loads from config.reward_model_name via AutoModelForCausalLM.from_pretrained.
   - Clips or squashes the final score into a reasonable range (default: sigmoid → [0,1] to match GSM8K reward scale). Expose this as a config.reward_score_activation option with values "sigmoid"|"tanh"|"none", default "sigmoid".

4. Blend logic: a BlendedScorer(RewardModelScorer) class that takes a learned scorer + the NoneReward scorer + an alpha, returns alpha * learned + (1 - alpha) * verifiable. build_reward_model should wrap the learned tier in BlendedScorer when config.reward_blend_alpha < 1.0 and ground_truths are available.

5. Add a simple `if __name__ == "__main__":` smoke at the bottom that builds a "none" scorer ONLY (no network call). Does NOT attempt to load a real HF model — leave the small/large tier exercise to Agent C's tests, which use a mock. The smoke prints a [B]-shaped tensor from the "none" tier on a hand-rolled batch and exits.

   Rationale: a real HF download in the __main__ smoke would fail on
   network-restricted CI runners and add minutes to local iterations.
   Agent C builds proper unit tests with `unittest.mock.MagicMock` for
   the `AutoModelForCausalLM` interface.

Also edit ppo_specs/config.py to add the six new fields documented in reward_model_integration.md §"Config knobs":
  reward_model_capacity: str = "none"
  reward_model_name: str | None = None
  reward_model_dtype: str = "auto"
  reward_model_reuse_reference: bool = False
  reward_blend_alpha: float = 1.0
  reward_score_activation: str = "sigmoid"

Also edit ppo_specs/checkpoint.py: update `_config_hash` (currently at
L210-L230). THREE of five RM fields are already in the `key_fields` dict
at L223-L225 (`reward_model_capacity`, `reward_blend_alpha`,
`reward_score_activation`). Add the TWO missing entries:
  "reward_model_name": getattr(config, "reward_model_name", None),
  "reward_model_reuse_reference": getattr(config, "reward_model_reuse_reference", False),

Use getattr with defaults so the hash is stable for older checkpoints.

Without this, resuming a run trained with a learned RM into a config with
capacity="none" will silently accept the mismatch and produce gibberish.

Do NOT modify ppo_specs/ppo_trainer.py in this task — that is Agent B's job. Do NOT create tests — that is Agent C's job.

Self-verification before you finish:
- python -c "from ppo_specs.reward_model import build_reward_model; import sys; print('ok')" succeeds
- python ppo_specs/reward_model.py (the __main__ block) prints a [B]-shaped tensor for the "none" tier
- grep 'reward_model' ppo_specs/config.py shows the six new fields
- grep 'reward_' ppo_specs/checkpoint.py shows the five new key_fields entries in _config_hash

Report: the three files you touched (config.py, checkpoint.py, reward_model.py), a one-sentence description of how NoneReward preserves pre-RM parity, and any deviations from the spec with justification.
```

---

## Agent B — Wire the RM into PPOTrainer

Subagent type: `general-purpose`

```
Task: Wire the learned reward model into the PPO trainer. Repo root: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. Assumes Agent A has landed ppo_specs/reward_model.py and the config additions.

Read first:
- ppo_specs/specs/reward_model_integration.md — §"File-level change list", §"Baseline invariant", §"Interaction with reward_mode", §"Adapter contract"
- ppo_specs/reward_model.py (newly created) — RewardModelScorer API
- ppo_specs/ppo_trainer.py L154-L244 (__init__), L267-L389 (generate_rollouts, esp. the per-sample reward call at L343), L954-L1035 (evaluate, esp. the reward call at L1024), L1040-L1197 (load_ppo_trainer)
- src/rewards.py L201-L285 — _RewardFnWrapper and make_reward_fn (Baseline invariant: do NOT delete or rewrite)

Make these edits in ppo_specs/ppo_trainer.py:

1. PPOTrainer.__init__ (L154-L244): add an OPTIONAL `reward_model_scorer: RewardModelScorer | None = None` keyword argument, store on self. Keep the existing `reward_fn` parameter — it stays as the primary path under the Baseline invariant. The new path is "if `self.reward_model_scorer is not None`, the trainer prefers the batched score_batch route in generate_rollouts and evaluate; otherwise it uses the existing per-sample self.reward_fn loop unchanged."

2. generate_rollouts (L267-L389): after the batched generate() call builds prompts, completions, and ground_truths, the per-sample `reward = self.reward_fn(completion, gt)` call lives at L343. Add a fast path: if `self.reward_model_scorer is not None`, replace the per-sample call with ONE batched call:
     rewards_t = self.reward_model_scorer.score_batch(local_prompts, completions_so_far, local_gts)
   Iterate the returned tensor to populate r.reward on each Rollout. Preserve the parse_success and format_match_boxed diagnostics (they still come from extract_answer_from_completion and matches_boxed_format — those are string-level and do not belong on the RM). The existing per-sample path remains for `reward_model_capacity == "none"` so today's behavior is bit-identical.

3. evaluate (L954-L1035): same fast-path substitution at L1024. After the batched generate() and decode, when `self.reward_model_scorer is not None`, call score_batch once and use the tensor as `mean_reward`. IMPORTANT: the existing `accuracy` metric treats any non-zero reward as "correct" — under a continuous RM, this is wrong. The returned accuracy MUST come from `gsm8k_reward` regardless of which scorer trained the policy. Either compute a parallel `gsm8k_reward` list locally inside `evaluate` for the accuracy return, or have the run scripts re-evaluate with the diagnostic_fn (the existing diagnostic-fn dance at run_e2_7.py:248-253 already does this for self_judge mode — extend the pattern).

4. load_ppo_trainer (L1040-L1197): construct the reward_model_scorer near the existing `make_reward_fn` call at L1173-L1175. If config.reward_model_reuse_reference is True and the reference_model exists, pass reference_model as base_model to build_reward_model. Otherwise let build_reward_model handle loading. Under Accelerate, call accelerator.unwrap_model before passing base_model. Re-use the `accelerator.main_process_first()` cache-warming pattern at L1079-L1099 / L1152-L1158 for the new RM download.

5. The existing `self.reward_fn = reward_fn` assignment: KEEP IT. Per the Baseline invariant, when `reward_model_capacity == "none"`, this is the only reward path. The new `self.reward_model_scorer` is additive, not a replacement.

6. Conflict-resolution check: if `config.reward_model_capacity != "none"` AND `config.reward_mode != "deterministic"`, raise ValueError early in load_ppo_trainer with the message:
     "reward_mode='{mode}' incompatible with reward_model_capacity='{cap}'; "
     "choose one continuous reward source (use reward_blend_alpha to mix "
     "RM with the deterministic verifier instead)."
   See the spec's "Conflict resolution" rule.

Edge cases to get right:
- Mock/no-RM parity: with config.reward_model_capacity="none", the new path must produce the SAME float values as the old per-sample loop. Stream a small hand-rolled test through mentally: 3 completions, 2 correct, 1 wrong — confirm you get {1.0, 1.0, 0.0} in the same order.
- Rank-0 gating: if any new print is added, guard it with accelerator.is_main_process when accelerator exists; else unconditional is fine.
- Frozen guarantee: add an assert right after loading the RM that no RM parameter has requires_grad=True. This catches future regressions.

Do NOT modify reward_model.py — that is Agent A's territory. Do NOT add tests — Agent C.

Self-verification:
- python -c "from ppo_specs.ppo_trainer import load_ppo_trainer, PPOTrainer; print('ok')" succeeds
- Diff should touch only ppo_specs/ppo_trainer.py
- grep 'self.reward_fn(' ppo_specs/ppo_trainer.py returns zero hits in the training-loop paths (ok to keep the attribute assignment)

Report: line ranges you changed, the three invariants you preserved (shape, device, float parity with gsm8k_reward on the "none" tier), and any API decisions that deviate from the spec.
```

---

## Agent C — Unit tests with mock RM

Subagent type: `general-purpose`

```
Task: Write unit tests for the new learned reward model integration. Repo: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. Assumes Agents A and B have landed.

Read first:
- ppo_specs/specs/reward_model_integration.md §"Unit tests"
- ppo_specs/tests/test_data_rewards.py — existing reward-test style, fixtures, and assertion conventions
- ppo_specs/tests/test_trainer.py — existing PPOTrainer integration-test style (mock model, mock tokenizer)
- ppo_specs/reward_model.py — the module you are testing
- ppo_specs/ppo_trainer.py — for integration test shape

Create ppo_specs/tests/test_reward_model.py with the five tests documented in the spec:

1. test_none_tier_parity — build_reward_model with capacity="none", call score_batch on a hand-rolled batch of 3 items (2 correct per gsm8k_reward rules, 1 wrong), assert element-wise float equality with calling gsm8k_reward directly.

2. test_small_tier_shape_dtype — mock AutoModelForCausalLM with a tiny fake model (create a minimal nn.Module that returns fake hidden states of the expected shape from a forward call). Do NOT download any real model — use monkeypatch on transformers.AutoModelForCausalLM.from_pretrained. Verify score_batch returns torch.Tensor of shape [B], dtype float32 (CPU path), device="cpu".

3. test_blend_alpha_interpolation — with alpha in {0.0, 0.5, 1.0}, confirm the blended output equals the expected convex combination of a mocked learned scorer (returns fixed tensor) and the gsm8k_reward values. Tolerance 1e-6.

4. test_reuse_reference_weight_sharing — construct a small shared base nn.Module, pass it as base_model to build_reward_model. Assert the returned scorer's base is the exact same Python object (id() equality or p.data_ptr() equality on the first parameter).

5. test_ground_truth_optional — with the mocked learned tier, passing ground_truths=None succeeds and still returns a valid tensor. With capacity="none", passing ground_truths=None raises ValueError with a clear message.

Additional test:
6. test_frozen_rm_params — after build_reward_model with capacity="small" (mocked), iterate scorer.parameters() and assert all have requires_grad=False. This enforces the §"Risks" invariant.

Constraints:
- Tests MUST run on CPU only (no torch.cuda usage). Use @pytest.mark.skipif(not torch.cuda.is_available(), ...) if a test genuinely needs GPU — it should not.
- No network access. All HF loads must be monkeypatched.
- Total runtime under 10 seconds on a laptop.
- Match the existing style in test_data_rewards.py: top-level TestXxx classes, descriptive docstrings, pytest conventions.

Self-verification:
- pytest ppo_specs/tests/test_reward_model.py -v — all 6 tests pass
- pytest ppo_specs/tests/ -v — existing tests still pass (no collateral damage)

Report: test count, total runtime, any tests that had to be adapted from the spec's plan (with reasons).
```

---

## Agent D1 — CPU smoke verification (no code changes)

Subagent type: `general-purpose`

```
Task: Verify the learned reward model integration works end-to-end on CPU. Repo: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. This is a verification task; make no code changes.

Read first:
- ppo_specs/specs/reward_model_integration.md §"Verification checklist" and §"Testing strategy"
- ppo_specs/config.py — find local_test_config()

Steps:

1. Run the "none" tier parity check (must be bitwise identical to pre-RM behavior):
     PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc
   Save the results/ppo_local_test.json.
   Re-run after setting an env var REWARD_MODEL_CAPACITY=none (if the run script exposes this; if not, construct a config with reward_model_capacity="none" explicitly).
   Diff the per-step reward, policy_loss, kl_divergence entries. Expect exact equality. Any drift is a regression — report it.

2. Run the "small" tier CPU smoke:
     PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc --reward-model-capacity small --reward-model-name <placeholder>
   If the run script does not yet expose these CLI flags, write a 10-line driver ppo_specs/scripts/smoke_rm.py that constructs the config in Python and calls run_e2_7 directly. Do NOT modify run_e2_7.py for this.
   For reward_model_name, use a small test HF model like Qwen/Qwen2.5-0.5B-Instruct (even though it is not a real RM, the value-head wrapper will still produce valid outputs — we are testing the plumbing, not reward quality).
   Expected: 5 steps complete, no OOM, no hang on RM forward.

3. Run the 4-process CPU DDP smoke if the Accelerate integration has landed:
     PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_cpu.yaml \
         ppo_specs/run_e2_7.py --local-test --no-mc --reward-model-capacity small --reward-model-name <placeholder>
   Expected: 4 processes; only rank 0 prints; gathered rewards are consistent across ranks; no hang on RM inference under unwrap_model.

Verify no unexpected side effects:
- results/ppo_local_test.json is present and well-formed in every run
- No new untracked files outside results/ and logs/
- No warnings about "CUDA out of memory" or "requires_grad" on frozen weights

Report:
- The three runs' final_acc values
- The diff between run 1 and run 1-bis (should be empty)
- Wall time for each run
- Anything unexpected (errors, warnings, memory pressure, hang timeouts)

Do NOT mark this task complete if any run fails or final_acc on the "none" tier doesn't match the baseline exactly.
```

---

## Agent D2 — Memory audit and profile (for the 8B path)

Subagent type: `general-purpose`

```
Task: Audit the memory behavior of the PPO pipeline with an 8B policy + 8B reference + 8B reward model. Repo: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. You will NOT run an 8B model — the audit is analytical plus a small-scale empirical check.

Read first:
- ppo_specs/specs/reward_model_integration.md §"Memory profile"
- ppo_specs/specs/memory_optimization.md
- ppo_specs/ppo_trainer.py L670-L738 (load_ppo_trainer) — all three model loads
- ppo_specs/reward_model.py — RM load path, especially the reuse_reference branch

Static analysis:

1. Trace every AutoModelForCausalLM.from_pretrained call in the PPO path and confirm:
   - The policy is the only one with requires_grad=True on any parameter
   - Reference and RM both have requires_grad=False on every parameter at load time
   - When reward_model_reuse_reference=True, only ONE from_pretrained call is made, not two
   Produce a table of (site, model name, trainable? yes/no, shared?).

2. Compute the expected peak memory for an 8B Llama-3 policy + 8B ref + 8B RM under:
   (a) no mitigations
   (b) + gradient_checkpointing on policy
   (c) + reward_model_reuse_reference=True
   (d) + manual CPU offload of frozen models during the K-epoch PPO update loop
   (e) + batch_size=8 (from 16)
   Use bf16 weights, bf16 grads, fp32 AdamW states (2x model size). Show your arithmetic.

3. Identify every torch.cuda.empty_cache() opportunity and every potential leak:
   - After RM scoring in generate_rollouts
   - Between capacities in run_e2_8.py-style sweeps
   - After checkpoint save/load
   Report any that are missing and recommend additions.

Empirical check (small-scale, runnable):

4. Run a 0.5B × 3 variant on whatever GPU you have (or on CPU if none). Command:
     PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --reward-model-capacity small --reward-model-name Qwen/Qwen2.5-0.5B-Instruct --reference-kl-coeff 0.01
   Before and after the first train_step, log:
     import torch; torch.cuda.max_memory_allocated() / 1e9
   (or psutil for CPU RSS if no GPU). Confirm the 0.5B × 3 peak matches the analytical estimate within ~20%.

5. Verify the reuse_reference path:
   Run the same smoke with reward_model_reuse_reference=True and confirm peak memory is ~1/3 less than without (since the RM base is not separately loaded).

Report:
- The analytical memory table (5 rows: a–e)
- The empirical peak from the 0.5B × 3 run, with and without reuse_reference
- Any regressions or surprises (>30% deviation from analytical estimate)
- A go/no-go recommendation for the 8B path on a single 80 GB A100, with the minimum set of mitigations required
- A list of code sites that should add gradient_checkpointing or empty_cache calls, with file:line
```

---

## Agent D3 — Final code review (superpowers:code-reviewer)

Subagent type: `superpowers:code-reviewer`

```
Task: Review the complete learned reward model integration for the PPO pipeline. Repo: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison.

The plan being executed is ppo_specs/specs/reward_model_integration.md. All implementation and tests have landed:
- NEW ppo_specs/reward_model.py
- MODIFIED ppo_specs/ppo_trainer.py
- MODIFIED ppo_specs/config.py
- NEW ppo_specs/tests/test_reward_model.py

Review against the spec's §"Verification checklist" and §"File-level change list". Focus areas:

1. Correctness of the "none" tier parity — does the new batched code path produce exactly the same reward values as the pre-change per-sample loop on a matched input? Read the implementation carefully — bitwise float parity is required.

2. Frozen-weight guarantee — every path that loads an RM or reference must end with requires_grad=False on every parameter AND eval() on the module. Check load_ppo_trainer and build_reward_model.

3. Rank-0 gating under Accelerate — any print, logger.save, or file write in the RM path must be gated on accelerator.is_main_process when an Accelerator is present. Check run_e2_7.py and run_e2_8.py touchpoints.

4. Score scale consistency — the RM output should be in a bounded range matching the GSM8K {0,1} scale, OR the advantage normalization in ppo_specs/advantage.py should be verified to handle unbounded continuous rewards. Read advantage.py to confirm.

5. Tokenizer independence — the LearnedRMScorer must re-tokenize from raw strings, not reuse the policy tokenizer's token ids from the Rollout, because the RM may use a different vocabulary. Check ppo_specs/reward_model.py.

6. Memory hygiene — reuse_reference=True must actually share weights (test: id() equality of base_model parameters). No duplicate AutoModelForCausalLM.from_pretrained when reuse is enabled.

7. Test coverage — are the six tests from the spec present, named consistently, all passing? Any holes (e.g., no test for the BlendedScorer edge cases at alpha=0 or alpha=1)?

8. Accuracy metric — now that rewards can be continuous, accuracy must be computed from gsm8k_reward separately, not from the RM output. Verify at all call sites in run_e2_7.py and run_e2_8.py.

9. Checkpoint hash — does _config_hash in ppo_specs/checkpoint.py include the new RM config fields? If not, a resume across different RM configs will silently accept a mismatch. Flag it.

10. Spec adherence — any silent deviations from reward_model_integration.md that weren't justified in the implementer's report?

Deliver a prioritized punch list:
- Blockers (must fix before merge)
- Non-blocking issues (cleanup, edge cases, tests to add)
- Nits (style, docstrings, comments)

For each item: file:line, a one-sentence description, and the recommended fix. Under 800 words.
```

---

## Agent D4 — GPU migration dry-run review

Subagent type: `general-purpose`

```
Task: Before the first 8B-model GPU cluster submission of the learned-RM PPO pipeline, do a pre-flight review. Repo: c:\Users\d81ru\Programming\MachineLearning\Project\RLVR-Comparison. Do NOT submit jobs; this is a review-only task.

Read first:
- ppo_specs/specs/reward_model_integration.md §"GPU migration test" and §"Verification checklist"
- Agent D2's memory audit report (if available in your context; otherwise flag its absence)
- scripts/slurm_e2_7.sh
- configs/accelerate_multi_gpu.yaml
- ppo_specs/config.py e2_7_config()

Confirm every item:

1. Memory safety:
   - reward_model_reuse_reference=True is set in the 8B config
   - gradient_checkpointing=True is set in the 8B config
   - batch_size is at most 8 (global) for the first 8B run
   - No _2x_ duplication of the 8B model anywhere in load_ppo_trainer

2. SLURM parameterization:
   - The submission command uses DEVICE_MODE=gpu and an accelerate config appropriate for multi-GPU
   - REWARD_MODEL_CAPACITY=large is propagated from submission to the Python config via env var or CLI
   - --gres=gpu:N matches NUM_GPUS matches accelerate num_processes

3. CPU smoke has been green:
   - Agent D1's three runs all passed
   - The "none" tier produced bitwise-identical metrics to the pre-RM baseline

4. First-run sentinel:
   - The config specifies a short n_steps (≤10) for the very first 8B submission
   - Eval frequency is at least every 5 steps so failure modes surface quickly
   - Checkpointing is enabled so a mid-run OOM does not lose signal
   - WandB or logger.save runs frequently enough that you can see the first-step metrics before OOM risk peaks

5. Rollback plan:
   - If the 8B run OOMs, what is the exact sequence of config changes to recover? (batch_size, gradient_checkpointing, offload, ZeRO — in order)

Produce a go/no-go sign-off with:
- Pass/fail for each of the 5 checks above
- The exact sbatch command to submit
- The exact rollback-on-OOM command
- A brief "monitor this for the first 10 minutes" list (what to watch in nvidia-smi and the logs)

Under 400 words. No code changes.
```

---

## Notes for the dispatcher

- **Serialize A and B** if the implementer is worried about merge conflicts in `ppo_specs/config.py`. A parallel variant: Agent A does config + reward_model.py; Agent B does ppo_trainer.py only and reads the spec for the config contract.
- **Worktrees**: consider dispatching A, B, C with `isolation: "worktree"` so each has a clean copy. D1–D4 are verification/review tasks — run them on the merged branch.
- **Don't dispatch D3 (code review) until D1 passes**. Reviewing code that hasn't been smoke-tested wastes cycles on issues that would show up in the smoke anyway.
- **Track context**: each agent starts from scratch. Re-paste the prompt; don't try to "continue from previous agent's work" — use explicit artifacts (files, test results) as the handoff medium.
- **Reporting**: agents' summaries land as your tool result. Relay the verification outcomes back to the developer; don't trust the summary — verify by reading the changed files and test output directly before declaring done.
