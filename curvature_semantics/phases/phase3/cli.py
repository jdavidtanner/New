"""CLI entry point for Phase 3."""

from __future__ import annotations

import argparse
import json

from curvature_semantics.core.config_manager import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Layerwise Decision Geometry")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output_root"] = args.output_dir

    mgr = ConfigManager(phase_config=args.config, overrides=overrides) if args.config else ConfigManager.from_phase(3, overrides=overrides)
    cfg = mgr.build()

    from curvature_semantics.phases.phase3.runner import run
    result = run(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
