"""Phase 2 orchestrator: loops over model registry, calls Phase 1 runner, aggregates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from curvature_semantics.core.config_manager import ConfigManager, ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.phases.phase1.runner import run as phase1_run
from curvature_semantics.phases.phase2.architecture_comparator import (
    compare_across_architectures, compute_effect_sizes, replication_summary,
)

logger = get_logger(__name__)


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_sweep = cfg.raw.get("model_sweep", [])
    registry = ConfigManager().get_model_registry()

    results_by_model: dict[str, pd.DataFrame] = {}

    for alias in model_sweep:
        logger.info("Running Phase 1 for model: %s", alias)
        if alias not in registry:
            logger.warning("Model alias '%s' not found in registry; skipping", alias)
            continue

        # Build a per-model Phase 1 config
        p1_config_path = cfg.raw.get("phase1_config", None)
        overrides = {
            "model": {"alias": alias},
            "phase": 2,
            "output_root": str(store.phase_dir / alias),
        }
        if p1_config_path:
            mgr = ConfigManager(phase_config=p1_config_path, overrides=overrides)
        else:
            mgr = ConfigManager.from_phase(1, overrides=overrides)
        p1_cfg = mgr.build()

        if cfg.use_cached:
            cached = ArtifactStore.from_config(p1_cfg).latest_run_dir(phase=1)
            if cached and (cached / "features.parquet").exists():
                logger.info("Using cached Phase 1 results for %s", alias)
                df = pd.read_parquet(cached / "features.parquet")
                results_by_model[alias] = df
                continue

        try:
            result = phase1_run(p1_cfg)
            p1_store = ArtifactStore.from_config(p1_cfg)
            df = p1_store.load_df("features")
            results_by_model[alias] = df
        except Exception as exc:
            logger.warning("Phase 1 failed for %s: %s", alias, exc)
            continue

    if not results_by_model:
        logger.error("No model results collected")
        return {"error": "no results"}

    # Aggregate
    all_df = pd.concat(results_by_model.values(), ignore_index=True)
    store.save_df("all_features", all_df)

    comparison = compare_across_architectures(results_by_model)
    effect_sizes = compute_effect_sizes(results_by_model, split_col="model_tuning")
    summary = replication_summary(results_by_model)

    store.save_json("architecture_comparison", comparison.to_dict())
    store.save_json("effect_sizes", effect_sizes)
    store.save_json("replication_summary", summary)

    logger.info("Phase 2 complete. Replication rate: %.2f", summary.get("replication_rate", 0))
    return {
        "n_models": len(results_by_model),
        "replication_summary": summary,
        "output_dir": str(store.phase_dir),
    }
