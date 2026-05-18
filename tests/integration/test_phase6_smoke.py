"""Smoke test: route examples through curvature router + all baselines."""

from __future__ import annotations

import numpy as np
import pytest

from curvature_semantics.phases.phase6.routing_policy import (
    ThresholdPolicy, RoutingSignal, RoutingAction,
)
from curvature_semantics.phases.phase6.curvature_router import CurvatureAwareRouter
from curvature_semantics.phases.phase6.baselines.vanilla_baseline import VanillaBaseline
from curvature_semantics.phases.phase6.baselines.rag_only_baseline import RAGOnlyBaseline
from curvature_semantics.phases.phase6.baselines.calibrated_baseline import CalibratedBaseline
from curvature_semantics.phases.phase6.baselines.self_consistency_baseline import SelfConsistencyBaseline
from curvature_semantics.phases.phase6.evaluator import evaluate_system, compare_systems


def dummy_generate(prompt: str) -> str:
    return f"Generated answer for: {prompt[:30]}"


def dummy_retrieve(prompt: str) -> str:
    return "Context: This is relevant background information."


class TestThresholdPolicy:
    def test_low_curvature_answers_directly(self):
        policy = ThresholdPolicy(low=0.3, medium=0.6, high=0.85)
        signal = RoutingSignal(curvature=0.1, completeness=0.9)
        assert policy.decide(signal) == RoutingAction.ANSWER

    def test_medium_curvature_retrieves(self):
        policy = ThresholdPolicy(low=0.3, medium=0.6, high=0.85)
        signal = RoutingSignal(curvature=0.5, completeness=0.5)
        assert policy.decide(signal) == RoutingAction.RETRIEVE

    def test_high_curvature_clarifies(self):
        policy = ThresholdPolicy(low=0.3, medium=0.6, high=0.85)
        signal = RoutingSignal(curvature=0.75, completeness=0.3)
        assert policy.decide(signal) == RoutingAction.CLARIFY

    def test_extreme_curvature_abstains(self):
        policy = ThresholdPolicy(low=0.3, medium=0.6, high=0.85)
        signal = RoutingSignal(curvature=0.99, completeness=0.0)
        assert policy.decide(signal) == RoutingAction.ABSTAIN


class TestCurvatureRouter:
    def test_route_low_curvature(self):
        router = CurvatureAwareRouter()
        signal = RoutingSignal(curvature=0.1, completeness=0.9)
        result = router.route("What is 2+2?", signal, dummy_generate, dummy_retrieve)
        assert result["action"] == "answer"
        assert "response" in result

    def test_route_high_curvature_abstains(self):
        router = CurvatureAwareRouter()
        signal = RoutingSignal(curvature=0.99, completeness=0.0)
        result = router.route("Highly uncertain question?", signal, dummy_generate, dummy_retrieve)
        assert result["action"] == "abstain"

    def test_estimate_signal_from_logits(self):
        router = CurvatureAwareRouter()
        logits = np.array([1.0, -1.0, 0.5, 2.0, -0.5])
        signal = router.estimate_signal(logits)
        assert 0.0 <= signal.curvature <= 1.0
        assert 0.0 <= signal.confidence <= 1.0


class TestBaselines:
    def test_vanilla(self):
        result = VanillaBaseline().run("test prompt", dummy_generate)
        assert result["action"] == "answer"
        assert len(result["response"]) > 0

    def test_rag_only(self):
        result = RAGOnlyBaseline().run("test prompt", dummy_generate, retrieve_fn=dummy_retrieve)
        assert result["action"] == "answer"
        assert "context" in result

    def test_calibrated_low_confidence_abstains(self):
        logits = np.zeros(1000)
        logits[0] = 0.01  # very flat distribution → low confidence
        result = CalibratedBaseline(confidence_threshold=0.9).run("test", dummy_generate, logits=logits)
        assert result["action"] == "abstain"

    def test_calibrated_high_confidence_answers(self):
        logits = np.zeros(1000)
        logits[0] = 100.0  # very peaked → high confidence
        result = CalibratedBaseline(confidence_threshold=0.5).run("test", dummy_generate, logits=logits)
        assert result["action"] == "answer"

    def test_self_consistency(self):
        result = SelfConsistencyBaseline(n_samples=3).run("test", dummy_generate)
        assert result["action"] == "answer"
        assert "consistency_score" in result


class TestEvaluator:
    def test_evaluate_system(self):
        results = [{"action": "answer", "response": "Paris is the capital."} for _ in range(5)]
        gold = ["Paris is the capital of France."] * 5
        metrics = evaluate_system(results, gold)
        assert "accuracy" in metrics
        assert "nli_entailment_mean" in metrics
        assert 0.0 <= metrics["coverage"] <= 1.0

    def test_compare_systems(self):
        systems = {
            "sys_a": [{"action": "answer", "response": "Paris"} for _ in range(5)],
            "sys_b": [{"action": "abstain", "response": "I don't know."} for _ in range(5)],
        }
        gold = ["Paris"] * 5
        df = compare_systems(systems, gold)
        assert "sys_a" in df.index
        assert "sys_b" in df.index
        assert "accuracy" in df.columns
