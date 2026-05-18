# Roadmap: Curvature as a Predictive Signal

## North-star goal

The goal of this project is to test whether **curvature-derived features from LLM hidden-state geometry predict semantic failure**.

"Moving forward" means moving closer to this predictive-validity question:

> Do curvature features predict incomplete, contradictory, uncertain, or incorrect answers on held-out examples, and do they add signal beyond standard baselines such as entropy, confidence, domain, layer, and model size?

## What the current POC establishes

The current proof-of-concept path validates software orchestration, not the scientific hypothesis. The `phase*_poc.yaml` configs and `mock-tiny` model show that all phases can run on CPU without downloads and can produce artifacts. That is the runway for real experiments.

## Next milestones

1. **POC CI gate**: keep the all-phase mock pipeline green so orchestration does not regress.
2. **Predictive report on POC artifacts**: run `curvature-predict` on Phase 1 artifacts to verify held-out prediction machinery.
3. **Small real-model smoke test**: run Phase 1/3 with a tiny real model and a small prompt set.
4. **Baseline-vs-curvature evaluation**: compare uncertainty/control-only predictors against predictors that include curvature features.
5. **Intervention validation**: test whether adding evidence or bridge concepts changes both curvature and semantic quality.
6. **Routing utility**: only after predictive lift exists, test whether curvature-aware routing improves real answer quality, abstention, or calibration.

## First predictive criterion

A useful first success criterion is:

> Adding curvature features improves held-out semantic-failure prediction over baseline controls.

Suggested metrics:

- AUROC / accuracy for binary semantic failure.
- R² / MAE for continuous semantic quality.
- Lift from baseline-only to baseline-plus-curvature models.

The implementation starts in `curvature_semantics.prediction.predictive_report` and is exposed as:

```bash
curvature-predict --data /path/to/features.parquet --output-dir results/prediction
```

or, if parquet support is unavailable:

```bash
curvature-predict --data /path/to/features.pkl --output-dir results/prediction
```
