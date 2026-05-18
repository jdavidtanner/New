"""Tests for config loading, Pydantic schema, and default merging."""

import pytest

from curvature_semantics.core.config_manager import ConfigManager, ExperimentConfig


class TestConfigManager:
    def test_load_defaults(self):
        mgr = ConfigManager()
        cfg = mgr.build()
        assert cfg.seed == 42
        assert cfg.batch_size == 8

    def test_cli_override(self):
        mgr = ConfigManager(overrides={"seed": 99, "batch_size": 16})
        cfg = mgr.build()
        assert cfg.seed == 99
        assert cfg.batch_size == 16

    def test_from_phase_1(self):
        mgr = ConfigManager.from_phase(1)
        cfg = mgr.build()
        assert cfg.phase == 1

    def test_from_phase_5(self):
        mgr = ConfigManager.from_phase(5)
        cfg = mgr.build()
        assert cfg.phase == 5

    def test_model_registry_populated(self):
        mgr = ConfigManager()
        registry = mgr.get_model_registry()
        assert len(registry) > 0
        first = list(registry.values())[0]
        assert "id" in first

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            ConfigManager.from_phase(99)

    def test_invalid_device_raises(self):
        with pytest.raises(Exception):
            ConfigManager(overrides={"device": "xpu"}).build()

    def test_model_alias_resolution(self):
        mgr = ConfigManager.from_phase(1)
        cfg = mgr.build()
        assert cfg.model.id != ""

    def test_to_dict_round_trip(self):
        cfg = ConfigManager(overrides={"seed": 77}).build()
        d = cfg.to_dict()
        assert d["seed"] == 77
