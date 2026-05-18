# Curvature Semantics

Curvature Semantics is an experimental research toolkit for evaluating the hypothesis that geometric obstruction in model hidden-state trajectories is associated with semantic incompleteness in language-model outputs.

The package is organized into six phases:

1. **Observational mapping**: extract hidden states, apply perturbations, compute curvature proxies and semantic completeness metrics, and correlate them.
2. **Cross-architecture replication**: repeat Phase 1 across a model sweep and summarize replication.
3. **Layerwise geometry**: profile curvature/completeness by transformer layer and region.
4. **Intervention testing**: add evidence, bridge concepts, contradiction removal, and retrieval interventions, then measure deltas.
5. **Synthetic ontology testing**: build controlled ontologies, remove/restore bridge concepts, train or simulate toy probes, and measure causal signatures.
6. **Atlas utility testing**: use curvature-aware routing and compare it against baseline answer/RAG/calibration/self-consistency systems.


## Project goal

The north-star goal is to determine whether curvature-derived features from LLM hidden-state geometry are predictive of semantic failure. In this repo, "moving forward" means getting closer to a held-out predictive test: do curvature features predict incomplete, contradictory, uncertain, or incorrect answers, and do they add signal beyond entropy, confidence, domain, layer, and model-size baselines? See `docs/ROADMAP.md` for the milestone plan.

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




## Controlled predictive calibration

Before spending GPU time, verify the evaluator can recover a known curvature signal:

```bash
FEATURES=$(python scripts/generate_controlled_features.py --output /tmp/controlled_features.parquet)
curvature-predict --data "$FEATURES" --output-dir /tmp/controlled_prediction
```

This is a calibration fixture, not evidence for the hypothesis. It should show positive lift for the curvature-enhanced model because the target is constructed from curvature features.


## Real-model smoke rung

When network/model downloads are available, run the same ladder against a tiny Hugging Face model:

```bash
python scripts/run_predictive_ladder.py \
  --phase1-config configs/phase1_real_smoke.yaml \
  --output-root /tmp/curvature_real_ladder
```

This is the first non-mock execution rung. It is still a smoke test, but it exercises real `transformers` model loading and hidden-state extraction.

## Moving toward the goal

The next executable rung is the predictive ladder: collect Phase 1 features, run a held-out baseline-vs-curvature predictive report, and optionally smoke-test the remaining POC phases.

```bash
python scripts/run_predictive_ladder.py --output-root /tmp/curvature_ladder --run-all-poc-phases
```

This command does not complete the scientific project; it tells us whether the software path from features to predictive-lift report is working. To make progress on the core hypothesis, replace `--phase1-config configs/phase1_poc.yaml` with a small real-model Phase 1 config and compare the reported baseline-vs-curvature lift.

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
