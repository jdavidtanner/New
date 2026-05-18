#!/usr/bin/env bash
# Run the full pipeline: Phase 1 → 2 → 3 → 4 → 5 → 6
set -euo pipefail

DEVICE=${DEVICE:-cuda}
OUTPUT_DIR=${OUTPUT_DIR:-results}
CONFIG_DIR=${CONFIG_DIR:-configs}

echo "=== Phase 1: Observational Mapping ==="
curvature-phase1 --config "$CONFIG_DIR/phase1_observational.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "=== Phase 2: Cross-Architecture Replication ==="
curvature-phase2 --config "$CONFIG_DIR/phase2_crossarch.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR" --use-cached

echo "=== Phase 3: Layerwise Decision Geometry ==="
curvature-phase3 --config "$CONFIG_DIR/phase3_layerwise.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "=== Phase 4: Intervention Testing ==="
curvature-phase4 --config "$CONFIG_DIR/phase4_intervention.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "=== Phase 5: Synthetic Ontology Test ==="
curvature-phase5 --config "$CONFIG_DIR/phase5_synthetic.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "=== Phase 6: Atlas Utility Test ==="
curvature-phase6 --config "$CONFIG_DIR/phase6_atlas.yaml" --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "=== Regression Report ==="
PHASE1_DATA=$(find "$OUTPUT_DIR/phase1" -name "features.parquet" | sort | tail -1)
if [[ -f "$PHASE1_DATA" ]]; then
    curvature-regress --data "$PHASE1_DATA" --output-dir "$OUTPUT_DIR/regression"
fi

echo "All phases complete. Results in $OUTPUT_DIR"
