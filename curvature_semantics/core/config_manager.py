"""Config loading: deep-merges defaults.yaml → models.yaml → phaseN.yaml → CLI overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    family: str = ""
    type: str = "dense"
    tuning: str = "base"
    params_billions: float = 0.0
    alias: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 42
    device: str = "cuda"
    batch_size: int = 8
    output_root: str = "results"
    log_level: str = "INFO"
    use_cached: bool = False
    num_workers: int = 4
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    num_generations: int = 5
    phase: int = 1
    name: str = "experiment"
    model: ModelSpec = Field(default_factory=ModelSpec)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid = {"cuda", "cpu", "mps"}
        if v not in valid:
            raise ValueError(f"device must be one of {valid}")
        return v

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return self.raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d.update(self.raw)
        return d


class ConfigManager:
    """Loads and merges YAML configs with CLI overrides."""

    _REPO_ROOT = Path(__file__).parent.parent.parent

    def __init__(
        self,
        phase_config: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        self._raw: dict[str, Any] = {}
        self._load_defaults()
        self._load_models()
        if phase_config is not None:
            self._load_yaml(Path(phase_config))
        if overrides:
            self._deep_merge(self._raw, overrides)

    # ------------------------------------------------------------------
    def _load_defaults(self) -> None:
        path = self._REPO_ROOT / "configs" / "defaults.yaml"
        if path.exists():
            self._load_yaml(path)

    def _load_models(self) -> None:
        path = self._REPO_ROOT / "configs" / "models.yaml"
        if path.exists():
            models_raw = self._read_yaml(path)
            self._raw.setdefault("_models_registry", {})
            for m in models_raw.get("models", []):
                alias = m.get("alias", m["id"])
                self._raw["_models_registry"][alias] = m

    def _load_yaml(self, path: Path) -> None:
        data = self._read_yaml(path)
        self._deep_merge(self._raw, data)

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    # ------------------------------------------------------------------
    def build(self) -> ExperimentConfig:
        raw = copy.deepcopy(self._raw)

        # Resolve model alias → full ModelSpec
        model_raw = raw.get("model", {})
        alias = model_raw.get("alias", "")
        registry = raw.get("_models_registry", {})
        if alias and alias in registry:
            resolved = {**registry[alias], **model_raw}
        else:
            resolved = model_raw
        raw["model"] = resolved

        cfg = ExperimentConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})
        cfg.raw = raw
        return cfg

    def get_model_registry(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._raw.get("_models_registry", {}))

    # ------------------------------------------------------------------
    @classmethod
    def from_phase(cls, phase: int, overrides: dict[str, Any] | None = None) -> "ConfigManager":
        phase_map = {
            1: "phase1_observational.yaml",
            2: "phase2_crossarch.yaml",
            3: "phase3_layerwise.yaml",
            4: "phase4_intervention.yaml",
            5: "phase5_synthetic.yaml",
            6: "phase6_atlas.yaml",
        }
        if phase not in phase_map:
            raise ValueError(f"Unknown phase: {phase}")
        path = cls._REPO_ROOT / "configs" / phase_map[phase]
        return cls(phase_config=path, overrides=overrides)
