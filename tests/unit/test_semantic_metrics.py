"""Unit tests for semantic metrics on canned examples."""

import pytest

from curvature_semantics.semantics.nli_scorer import NLIScorer
from curvature_semantics.semantics.factuality_scorer import FactualityScorer
from curvature_semantics.semantics.self_consistency import SelfConsistency
from curvature_semantics.semantics.calibration import CalibrationError
from curvature_semantics.semantics.ood_uncertainty import OODUncertainty
from curvature_semantics.semantics.abstention_accuracy import AbstentionAccuracy
from curvature_semantics.semantics.evidence_utilization import EvidenceUtilization
from curvature_semantics.semantics.answer_completeness import AnswerCompleteness
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
import numpy as np


class TestNLIScorer:
    def test_returns_expected_keys(self):
        scorer = NLIScorer()
        result = scorer.score("The sky is blue.", "The sky is blue.", context="The sky is blue.")
        assert "nli_entailment" in result
        assert "nli_contradiction" in result
        assert "nli_neutral" in result

    def test_scores_sum_to_one(self):
        scorer = NLIScorer()
        result = scorer.score("A", "B", context="A")
        total = result["nli_entailment"] + result["nli_contradiction"] + result["nli_neutral"]
        assert abs(total - 1.0) < 1e-4

    def test_empty_response(self):
        scorer = NLIScorer()
        result = scorer.score("Q", "", context="C")
        assert result["nli_entailment"] == 0.0

    def test_batch_score_length(self):
        scorer = NLIScorer()
        pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        results = scorer.batch_score(pairs)
        assert len(results) == 3


class TestFactualityScorer:
    def test_returns_factuality_key(self):
        scorer = FactualityScorer()
        result = scorer.score("Q", "The answer is yes.", context="The answer is yes.")
        assert "factuality" in result
        assert 0.0 <= result["factuality"] <= 1.0

    def test_no_context_returns_neutral(self):
        scorer = FactualityScorer()
        result = scorer.score("Q", "Some response.", context="")
        assert result["factuality"] == 0.5


class TestSelfConsistency:
    def test_identical_responses(self):
        sc = SelfConsistency()
        result = sc.score("Q", "A", generations=["A", "A", "A"])
        assert result["self_consistency"] > 0.9

    def test_single_generation(self):
        sc = SelfConsistency()
        result = sc.score("Q", "A", generations=["A"])
        assert result["self_consistency"] == 1.0

    def test_diverse_responses_lower_consistency(self):
        sc = SelfConsistency()
        diverse = ["cats are mammals", "photosynthesis is a process", "water is H2O", "the moon orbits earth"]
        similar = ["cats are mammals", "cats are animals", "cats are pets", "cats have fur"]
        r_diverse = sc.score("Q", "", generations=diverse)
        r_similar = sc.score("Q", "", generations=similar)
        assert r_diverse["self_consistency"] <= r_similar["self_consistency"] + 0.1


class TestCalibrationError:
    def test_perfect_calibration(self):
        cal = CalibrationError()
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 10)
        correctness = np.array([0, 0, 1, 1, 1] * 10, dtype=float)
        result = cal.compute_ece(confidences, correctness)
        assert "calibration_error" in result
        assert 0.0 <= result["calibration_error"] <= 1.0

    def test_single_score(self):
        cal = CalibrationError()
        result = cal.score("", "", confidence=0.9, correct=True)
        assert result["calibration_error"] < 0.2

    def test_overconfident_single(self):
        cal = CalibrationError()
        result = cal.score("", "", confidence=0.9, correct=False)
        assert result["calibration_error"] > 0.5


class TestOODUncertainty:
    def test_from_logits(self):
        scorer = OODUncertainty()
        logits = np.array([1.0, 0.5, -0.5, 2.0])
        result = scorer.score("Q", "A", logits=logits)
        assert "entropy" in result
        assert result["entropy"] >= 0.0

    def test_from_token_logprobs(self):
        logprobs = np.array([-0.5, -1.0, -0.8, -0.3])
        result = OODUncertainty.compute_from_token_logprobs(logprobs)
        assert "perplexity" in result
        assert result["perplexity"] > 1.0


class TestAbstentionAccuracy:
    def test_detects_abstention_phrases(self):
        aa = AbstentionAccuracy()
        assert aa.detect_abstention("I don't know the answer.")
        assert aa.detect_abstention("I'm not sure about this.")
        assert aa.detect_abstention("I cannot determine that.")
        assert not aa.detect_abstention("The capital of France is Paris.")

    def test_corpus_metrics(self):
        preds = [{"response": "I don't know."}, {"response": "Paris is the capital."}, {"response": "I'm uncertain."}]
        labels = [True, False, True]
        result = AbstentionAccuracy.compute_corpus_metrics(preds, labels)
        assert "abstention_f1" in result
        assert 0.0 <= result["abstention_f1"] <= 1.0


class TestEvidenceUtilization:
    def test_perfect_overlap(self):
        eu = EvidenceUtilization()
        result = eu.score("Q", "cats are mammals with fur", context="cats are mammals with fur")
        assert result["evidence_utilization"] > 0.9

    def test_zero_overlap(self):
        eu = EvidenceUtilization()
        result = eu.score("Q", "completely different words here", context="apples oranges bananas fruit")
        assert result["evidence_utilization"] < 0.3

    def test_empty_context(self):
        eu = EvidenceUtilization()
        result = eu.score("Q", "some response", context="")
        assert result["evidence_utilization"] == 0.0


class TestAnswerCompleteness:
    def test_with_reference(self):
        ac = AnswerCompleteness()
        result = ac.score("Q", "Paris is the capital of France", reference="Paris is the capital of France")
        assert result["answer_completeness"] > 0.8

    def test_empty_reference(self):
        ac = AnswerCompleteness()
        result = ac.score("Q", "some answer", reference="")
        assert result["answer_completeness"] == 0.0


class TestCompletenessAggregator:
    def test_basic_scoring(self):
        agg = CompletenessAggregator(metric_names=["nli_entailment", "self_consistency"])
        bundle = agg.score("Q", "A", context="A", domain="test")
        assert "nli_entailment" in bundle.metrics
        assert bundle.primary() >= 0.0

    def test_to_flat_dict(self):
        agg = CompletenessAggregator(metric_names=["evidence_utilization"])
        bundle = agg.score("Q", "cats", context="cats are animals", domain="animals")
        d = bundle.to_flat_dict()
        assert "domain" in d
        assert "evidence_utilization" in d
