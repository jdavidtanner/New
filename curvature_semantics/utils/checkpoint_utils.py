"""Deterministic checkpoint naming and resume logic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any], n_chars: int = 8) -> str:
    """Deterministic short hash of a config dict."""
    serialized = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()[:n_chars]


def find_latest_checkpoint(directory: Path, prefix: str = "ckpt") -> Path | None:
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob(f"{prefix}_*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def checkpoint_path(directory: Path, step: int, prefix: str = "ckpt") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{step:06d}.pt"
