# Curvature Semantics

Curvature Semantics is an experimental research toolkit for evaluating the hypothesis that geometric obstruction in model hidden-state trajectories is associated with semantic incompleteness in language-model outputs.

The package is organized into six phases:

1. **Observational mapping**: extract hidden states, apply perturbations, compute curvature proxies and semantic completeness metrics, and correlate them.
2. **Cross-architecture replication**: repeat Phase 1 across a model sweep and summarize replication.
3. **Layerwise geometry**: profile curvature/completeness by transformer layer and region.
4. **Intervention testing**: add evidence, bridge concepts, contradiction removal, and retrieval interventions, then measure deltas.
5. **Synthetic ontology testing**: build controlled ontologies, remove/restore bridge concepts, train or simulate toy probes, and measure causal signatures.
6. **Atlas utility testing**: use curvature-aware routing and compare it against baseline answer/RAG/calibration/self-consistency systems.

## Installation

```bash
pip install -e .
```

Optional extras enable heavier metrics and experiments:

```bash
pip install -e '.[semantics,curvature,tda,regression,viz,finetune,dev]'
```

The code has dependency-free fallbacks for several development paths, but full scientific runs should install the appropriate optional extras.

## Quick checks

```bash
python -m pytest tests/unit
python -m compileall -q curvature_semantics tests
```


## Proof-of-concept run

The repository includes download-free proof-of-concept configs that use the built-in `mock-tiny` model and synthetic datasets. These are intended to validate orchestration for every phase on a CPU-only machine:

```bash
for p in 1 2 3 4 5 6; do
  python -m curvature_semantics.phases.phase${p}.cli \
    --config configs/phase${p}_poc.yaml \
    --output-dir /tmp/curvature_poc
done
```

These POC runs are not scientifically meaningful model evaluations; they are smoke tests for data flow, artifact generation, curvature/semantic metric computation, and phase-to-phase orchestration.

## CLI entry points

```bash
curvature-phase1 --config configs/phase1_observational.yaml --device cpu
curvature-phase2 --config configs/phase2_crossarch.yaml --device cpu
curvature-phase3 --config configs/phase3_layerwise.yaml --device cpu
curvature-phase4 --config configs/phase4_intervention.yaml --device cpu
curvature-phase5 --config configs/phase5_quick.yaml --device cpu
curvature-phase6 --config configs/phase6_atlas.yaml --device cpu
```
