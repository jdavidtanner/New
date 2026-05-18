"""CLI entry point for Phase 1."""

from __future__ import annotations

import argparse
import json

from curvature_semantics.core.config_manager import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Observational Mapping")
    parser.add_argument("--config", default=None, help="Path to phase1 YAML config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-cached", action="store_true")
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output_root"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.use_cached:
        overrides["use_cached"] = True

    mgr = ConfigManager(phase_config=args.config, overrides=overrides) if args.config else ConfigManager.from_phase(1, overrides=overrides)
    cfg = mgr.build()

    from curvature_semantics.phases.phase1.runner import run
    result = run(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
