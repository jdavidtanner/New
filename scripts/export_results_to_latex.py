#!/usr/bin/env python
"""Read artifact store; render regression tables and figures for paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results to LaTeX")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-dir", default="paper_assets", help="Where to write .tex and figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect Phase 1 regression results
    reg_files = list(results_dir.glob("**/report.json"))
    for f in reg_files:
        data = json.loads(f.read_text())
        ols = data.get("ols", {})
        r2 = ols.get("r_squared", "N/A")
        print(f"Phase 1 OLS R² = {r2:.4f}" if isinstance(r2, float) else f"R² = {r2}")

    # Phase 5 causal test
    causal_files = list(results_dir.glob("**/phase5_summary.json"))
    for f in causal_files:
        data = json.loads(f.read_text())
        print(f"Phase 5 hypothesis supported: {data.get('hypothesis_supported')}")
        print(f"  Curvature elevation (without bridges): {data.get('avg_curvature_elevation_without_bridges', 'N/A'):.4f}")
        print(f"  Curvature elevation (with bridges):    {data.get('avg_curvature_elevation_with_bridges', 'N/A'):.4f}")

    # Phase 6 comparison
    comp_files = list(results_dir.glob("**/system_comparison.json"))
    for f in comp_files:
        data = json.loads(f.read_text())
        df = pd.DataFrame(data)
        print("\nSystem comparison:")
        print(df.to_string())
        (output_dir / "system_comparison.tex").write_text(df.to_latex(float_format="%.3f"))
        print(f"Written to {output_dir}/system_comparison.tex")


if __name__ == "__main__":
    main()
