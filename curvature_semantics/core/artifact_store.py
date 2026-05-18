"""Structured read/write for experiment artifacts: arrays, DataFrames, JSON, checkpoints."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactStore:
    """Manages experiment artifact directories with deterministic naming."""

    def __init__(self, output_root: str | Path, phase: int, run_id: str | None = None):
        self.root = Path(output_root)
        self.phase = phase
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.phase_dir = self.root / f"phase{phase}" / self.run_id
        self.phase_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.phase_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    def save_array(self, name: str, arr: np.ndarray, subdir: str = "") -> Path:
        p = self.path(subdir, f"{name}.npy") if subdir else self.path(f"{name}.npy")
        np.save(p, arr)
        return p

    def load_array(self, name: str, subdir: str = "") -> np.ndarray:
        p = self.path(subdir, f"{name}.npy") if subdir else self.path(f"{name}.npy")
        return np.load(p, allow_pickle=False)

    # ------------------------------------------------------------------
    def save_df(self, name: str, df: pd.DataFrame, subdir: str = "") -> Path:
        p = self.path(subdir, f"{name}.parquet") if subdir else self.path(f"{name}.parquet")
        try:
            df.to_parquet(p, index=True)
            return p
        except Exception:
            fallback = p.with_suffix(".pkl")
            df.to_pickle(fallback)
            return fallback

    def load_df(self, name: str, subdir: str = "") -> pd.DataFrame:
        p = self.path(subdir, f"{name}.parquet") if subdir else self.path(f"{name}.parquet")
        if p.exists():
            return pd.read_parquet(p)
        fallback = p.with_suffix(".pkl")
        if fallback.exists():
            return pd.read_pickle(fallback)
        return pd.read_parquet(p)

    # ------------------------------------------------------------------
    def save_json(self, name: str, data: Any, subdir: str = "") -> Path:
        p = self.path(subdir, f"{name}.json") if subdir else self.path(f"{name}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return p

    def load_json(self, name: str, subdir: str = "") -> Any:
        p = self.path(subdir, f"{name}.json") if subdir else self.path(f"{name}.json")
        with open(p) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def save_pickle(self, name: str, obj: Any, subdir: str = "") -> Path:
        p = self.path(subdir, f"{name}.pkl") if subdir else self.path(f"{name}.pkl")
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        return p

    def load_pickle(self, name: str, subdir: str = "") -> Any:
        p = self.path(subdir, f"{name}.pkl") if subdir else self.path(f"{name}.pkl")
        with open(p, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    def save_config(self, config_dict: dict[str, Any]) -> Path:
        return self.save_json("config", config_dict)

    def exists(self, name: str, ext: str = "npy", subdir: str = "") -> bool:
        p = self.path(subdir, f"{name}.{ext}") if subdir else self.path(f"{name}.{ext}")
        return p.exists()

    def latest_run_dir(self, phase: int | None = None) -> Path | None:
        """Return the most recent run directory for a phase."""
        phase = phase or self.phase
        phase_root = self.root / f"phase{phase}"
        if not phase_root.exists():
            return None
        runs = sorted(phase_root.iterdir(), key=lambda p: p.name)
        return runs[-1] if runs else None

    @classmethod
    def from_config(cls, cfg: Any) -> ArtifactStore:
        return cls(
            output_root=cfg.output_root,
            phase=cfg.phase,
        )
