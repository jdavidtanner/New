#!/usr/bin/env python3
"""Run the first executable rungs toward predictive validation.

This script is intentionally small and conservative: it does not claim to complete the
scientific project. It automates the current next step, which is to produce a Phase 1 feature
artifact and immediately ask whether curvature features add held-out predictive signal over
baseline controls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_PHASE1_CONFIG = "configs/phase1_poc.yaml"
DEFAULT_OUTPUT_ROOT = "results/predictive_ladder"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 plus held-out curvature predictive evaluation."
    )
    parser.add_argument("--phase1-config", default=DEFAULT_PHASE1_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-col", default="nli_entailment")
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--split-col", default="prompt")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-all-poc-phases",
        action="store_true",
        help="Also run phase2_poc.yaml through phase6_poc.yaml as orchestration smoke tests.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "goal": (
            "Move toward the north-star question: do curvature features predict semantic "
            "failure on held-out examples beyond baseline controls?"
        ),
        "phase1_config": args.phase1_config,
        "output_root": str(output_root),
        "steps": [],
    }

    _run_step(
        [
            sys.executable,
            "-m",
            "curvature_semantics.phases.phase1.cli",
            "--config",
            args.phase1_config,
            "--output-dir",
            str(output_root),
        ],
        status,
        "phase1_feature_collection",
    )

    features_path = _latest_feature_artifact(output_root)
    prediction_dir = output_root / "prediction"
    _run_step(
        [
            sys.executable,
            "-m",
            "curvature_semantics.prediction.predictive_report",
            "--data",
            str(features_path),
            "--output-dir",
            str(prediction_dir),
            "--target-col",
            args.target_col,
            "--failure-threshold",
            str(args.failure_threshold),
            "--split-col",
            args.split_col,
            "--test-fraction",
            str(args.test_fraction),
            "--seed",
            str(args.seed),
        ],
        status,
        "held_out_predictive_report",
    )

    report_path = prediction_dir / "predictive_report.json"
    report = json.loads(report_path.read_text())
    status["predictive_report"] = str(report_path)
    status["predictive_interpretation"] = report.get("interpretation")
    status["predictive_lift"] = report.get("lift", {})

    if args.run_all_poc_phases:
        for phase in range(2, 7):
            _run_step(
                [
                    sys.executable,
                    "-m",
                    f"curvature_semantics.phases.phase{phase}.cli",
                    "--config",
                    f"configs/phase{phase}_poc.yaml",
                    "--output-dir",
                    str(output_root),
                ],
                status,
                f"phase{phase}_poc_smoke",
            )

    status_path = output_root / "predictive_ladder_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))
    print(f"\nStatus written to: {status_path}")


def _run_step(command: list[str], status: dict[str, Any], name: str) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    step = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    status["steps"].append(step)
    if completed.stdout:
        print(completed.stdout[-4000:])
    if completed.stderr:
        print(completed.stderr[-4000:], file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {name}")


def _latest_feature_artifact(output_root: Path) -> Path:
    candidates = list(output_root.glob("phase1/*/features.parquet"))
    candidates.extend(output_root.glob("phase1/*/features.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No Phase 1 features artifact found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
