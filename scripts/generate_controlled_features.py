#!/usr/bin/env python3
"""Generate a controlled feature table with known curvature predictive signal.

This is not a scientific experiment. It is a calibration fixture for the predictive evaluator:
curvature features are constructed to predict semantic completeness while baseline controls carry
little signal. A healthy predictive report should show positive lift for baseline+curvature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled curvature feature data")
    parser.add_argument("--output", default="/tmp/controlled_features.parquet")
    parser.add_argument("--n-prompts", type=int, default=80)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_controlled_features(n_prompts=args.n_prompts, layers=args.layers, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output, index=False)
    except Exception:
        output = output.with_suffix(".pkl")
        df.to_pickle(output)
    print(output)


def make_controlled_features(n_prompts: int = 80, layers: int = 4, seed: int = 42) -> pd.DataFrame:
    """Return features where curvature, but not baseline controls, predicts completeness."""
    rng = np.random.default_rng(seed)
    rows = []
    for prompt_idx in range(n_prompts):
        prompt_curvature = rng.uniform(0.0, 1.0)
        domain = "controlled_easy" if prompt_idx < n_prompts // 2 else "controlled_hard"
        for layer_idx in range(layers):
            layer_bump = 0.04 * layer_idx
            curvature = np.clip(prompt_curvature + layer_bump + rng.normal(0.0, 0.02), 0.0, 1.2)
            entailment = np.clip(0.9 - 0.65 * curvature + rng.normal(0.0, 0.04), 0.0, 1.0)
            rows.append(
                {
                    "prompt": f"controlled-prompt-{prompt_idx}",
                    "domain": domain,
                    "layer_idx": layer_idx,
                    "entropy": rng.uniform(0.2, 0.8),
                    "confidence": rng.uniform(0.4, 0.9),
                    "model_params_billions": 0.0001,
                    "trajectory_divergence": float(curvature),
                    "intrinsic_dimension": float(1.0 + 2.0 * curvature),
                    "neighborhood_distortion": float(0.2 * curvature),
                    "nli_entailment": float(entailment),
                    "nli_contradiction": float(max(0.0, 1.0 - entailment) * 0.5),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
