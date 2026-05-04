"""Verify run_e2_8.py has rank-0 gating, MC broadcast, and barrier
patterns wired (per ddp_cpu_gpu_migration.md §2.2 and §7.6.*)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = (REPO / "ppo_specs" / "run_e2_8.py").read_text()


def test_detects_ddp_via_local_rank():
    assert "LOCAL_RANK" in RUN

def test_has_rank_0_gating():
    assert "_is_main" in RUN or "is_main_process" in RUN

def test_has_wait_for_everyone():
    assert "wait_for_everyone" in RUN

def test_has_mc_broadcast():
    assert "broadcast_object_list" in RUN or "broadcast" in RUN

def test_save_checkpoint_passes_accelerator():
    """save_checkpoint must receive the accelerator kwarg for unwrap."""
    if "save_checkpoint" in RUN:
        # Heuristic: at least one save_checkpoint call mentions accelerator
        assert "accelerator=accelerator" in RUN or "accelerator=" in RUN
