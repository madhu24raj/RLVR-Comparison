"""Tests for memory-optimization config knobs and checkpoint resume fix."""
import sys
import os
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig


class TestNewConfigFields:
    def test_optimizer_8bit_default_false(self):
        cfg = PPOConfig()
        assert cfg.optimizer_8bit is False

    def test_optimizer_fused_default_false(self):
        cfg = PPOConfig()
        assert cfg.optimizer_fused is False

    def test_reference_quant_default_none(self):
        cfg = PPOConfig()
        assert cfg.reference_quant == "none"

    def test_reference_quant_valid_values(self):
        for v in ("none", "int8", "nf4"):
            cfg = PPOConfig(reference_quant=v)
            assert cfg.reference_quant == v

    def test_length_bucketed_generation_default_false(self):
        cfg = PPOConfig()
        assert cfg.length_bucketed_generation is False
        assert cfg.generation_bucket_size == 4

    def test_all_new_fields_serializable(self):
        """Config dataclass must roundtrip through asdict() for checkpointing."""
        from dataclasses import asdict
        cfg = PPOConfig(
            optimizer_8bit=True, optimizer_fused=True,
            reference_quant="int8", length_bucketed_generation=True,
            generation_bucket_size=8,
        )
        d = asdict(cfg)
        assert d["optimizer_8bit"] is True
        assert d["reference_quant"] == "int8"
        assert d["generation_bucket_size"] == 8


class TestCheckpointMapLocation:
    """Verify checkpoint loaders use CPU map_location to avoid 2x memory peak."""

    def test_checkpoint_module_imports(self):
        from ppo_specs import checkpoint
        assert hasattr(checkpoint, "load_checkpoint")

    def test_load_checkpoint_uses_cpu_map_location(self):
        """The source code of load_checkpoint must use map_location='cpu'
        for the three torch.load calls that load optimizer/critic state."""
        import inspect
        from ppo_specs import checkpoint
        src = inspect.getsource(checkpoint.load_checkpoint)
        # Three optimizer/critic loads; all must be cpu-mapped
        cpu_count = src.count('map_location="cpu"') + src.count("map_location='cpu'")
        # rng_states load is already CPU-mapped, so we expect at least 4
        assert cpu_count >= 4, (
            f"Expected >=4 'map_location=\"cpu\"' calls in load_checkpoint "
            f"to prevent 2x optimizer-state GPU peak; found {cpu_count}.\n"
            f"Source:\n{src}"
        )
        # And no map_location=device for the optimizer/critic loads
        assert "map_location=device" not in src, (
            "load_checkpoint must NOT load optimizer/critic state to GPU "
            "(map_location=device) — this causes a 2x transient peak. "
            "Use map_location='cpu' instead."
        )
