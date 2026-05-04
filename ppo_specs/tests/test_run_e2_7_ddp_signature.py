"""Verify run_e2_7.py has rank-0 gating, exit broadcast, MC broadcast,
and re-prepare-on-resume patterns wired in (per ddp_cpu_gpu_migration.md
§7.6.*)."""
import inspect
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = (REPO / "ppo_specs" / "run_e2_7.py").read_text(encoding="utf-8")


def _scan_save_checkpoint_barrier_coverage(path: Path):
    """Returns list of (line_num, line_text) for save_checkpoint calls
    that lack a preceding _wait() / wait_for_everyone() within 10 lines.

    Enforces the DDP-BAR contract from ddp_cpu_gpu_migration.md §2.1:
    every save_checkpoint(...) call site MUST be preceded by a barrier
    reaching every rank, because checkpoint.save_checkpoint() is
    caller-gated (see checkpoint.py:50-63 docstring) and putting the
    barrier inside the function would deadlock non-rank-0 ranks.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    violations = []
    for i, line in enumerate(lines):
        if re.search(r"save_checkpoint\s*\(", line):
            window = lines[max(0, i - 10):i]
            if not any(
                re.search(r"_wait\s*\(\)|wait_for_everyone\s*\(\)", w)
                for w in window
            ):
                violations.append((i + 1, line))
    return violations


def test_imports_accelerator_optionally():
    """Should import Accelerator inside the LOCAL_RANK branch
    or at the top with a graceful fallback."""
    assert "Accelerator" in RUN or "accelerator" in RUN


def test_detects_ddp_via_local_rank():
    assert "LOCAL_RANK" in RUN, (
        "run_e2_7 must auto-detect DDP via the LOCAL_RANK env var."
    )


def test_has_rank_0_gating():
    """Every save_checkpoint and logger.save call must be gated."""
    # heuristic: a `_is_main()` or `accelerator.is_main_process` somewhere
    assert "_is_main" in RUN or "is_main_process" in RUN


def test_has_wait_for_everyone_around_save():
    assert "wait_for_everyone" in RUN, (
        "Must call accelerator.wait_for_everyone() before save_checkpoint "
        "(§7.6 critical hazard: rank-0 can deadlock without barrier)."
    )


def test_has_mc_broadcast():
    """MC baselines must be computed on rank 0 and broadcast (§7.6.5)."""
    assert "broadcast_object_list" in RUN or "broadcast" in RUN


def test_has_exit_signal_reduce():
    """§7.6.4: exit signal must use accelerator.reduce so single-rank
    SIGTERM doesn't deadlock other ranks."""
    assert "exit_flag" in RUN or "should_exit_all" in RUN, (
        "Exit handler must broadcast/reduce the should_exit flag across ranks."
    )


def test_has_set_questions_local_shard():
    """§7.6.2: set_questions must be called on the LOCAL shard, not the
    global batch, otherwise self-judge questions misalign."""
    # heuristic: source contains a _shard_list-style call near set_questions
    if "set_questions" in RUN:
        assert "_shard_list" in RUN or "process_index" in RUN, (
            "set_questions must use the local shard under DDP (§7.6.2)."
        )


def test_resume_re_prepares_under_ddp():
    """§7.6.3: after reassigning trainer.model on resume, must call
    accelerator.prepare again or the wrapping is lost."""
    if "from_pretrained" in RUN and "policy_optimizer.load_state_dict" in RUN:
        # A re-prepare must appear after the reassignment
        assert "accelerator.prepare" in RUN, (
            "After resume reassigns trainer.model, must re-call "
            "accelerator.prepare to restore DDP wrapping (§7.6.3)."
        )


def test_save_checkpoint_barrier_coverage_run_e2_7():
    """DDP-BAR regression guard: every save_checkpoint(...) call in
    run_e2_7.py must be preceded by a _wait() or wait_for_everyone()
    barrier within the previous 10 lines. Catches future regressions
    cheaply at the string level."""
    path = REPO / "ppo_specs" / "run_e2_7.py"
    violations = _scan_save_checkpoint_barrier_coverage(path)
    assert not violations, (
        "save_checkpoint calls without preceding _wait() / "
        f"wait_for_everyone() within 10 lines in run_e2_7.py: {violations}"
    )


def test_save_checkpoint_barrier_coverage_run_e2_8():
    """DDP-BAR regression guard for run_e2_8.py. E2.8 has no
    save_checkpoint calls today, so this passes vacuously; when E2.8
    adds checkpointing, it must mirror the run_e2_7 barrier pattern."""
    path = REPO / "ppo_specs" / "run_e2_8.py"
    violations = _scan_save_checkpoint_barrier_coverage(path)
    assert not violations, (
        "save_checkpoint calls without preceding _wait() / "
        f"wait_for_everyone() within 10 lines in run_e2_8.py: {violations}"
    )
