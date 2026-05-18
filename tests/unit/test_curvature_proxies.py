"""Unit tests for curvature proxies on synthetic tensors."""

import numpy as np
import pytest

from curvature_semantics.curvature.trajectory_divergence import LocalTrajectoryDivergence
from curvature_semantics.curvature.intrinsic_dimension import IntrinsicDimension
from curvature_semantics.curvature.neighborhood_distortion import NeighborhoodDistortionRatio
from curvature_semantics.curvature.geodesic_deviation import GeodesicDeviationProxy
from curvature_semantics.curvature.ricci_graph_curvature import RicciGraphCurvature
from curvature_semantics.curvature.persistent_homology import PersistentHomologyFeatures
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator


class TestLocalTrajectoryDivergence:
    def test_basic_output(self, hidden_states_2d):
        proxy = LocalTrajectoryDivergence(n_neighbors=5)
        result = proxy.compute(hidden_states_2d)
        assert "trajectory_divergence" in result
        assert isinstance(result["trajectory_divergence"], float)
        assert result["trajectory_divergence"] >= 0.0

    def test_single_sample_returns_zero(self):
        proxy = LocalTrajectoryDivergence()
        result = proxy.compute(np.array([[1.0, 2.0, 3.0]]))
        assert result["trajectory_divergence"] == 0.0

    def test_trajectory_pair(self, rng):
        original = rng.standard_normal((8, 64)).astype(np.float32)
        perturbed = original + rng.standard_normal((8, 64)).astype(np.float32) * 0.1
        proxy = LocalTrajectoryDivergence()
        result = proxy.compute_from_trajectory_pair(original, perturbed)
        assert "trajectory_divergence" in result
        assert "trajectory_divergence_final" in result
        assert result["trajectory_divergence"] >= 0.0

    def test_identical_trajectory_divergence_low(self, rng):
        x = rng.standard_normal((8, 64)).astype(np.float32)
        result = LocalTrajectoryDivergence().compute_from_trajectory_pair(x, x.copy())
        assert result["trajectory_divergence"] < 0.01

    def test_high_perturbation_increases_divergence(self):
        rng = np.random.default_rng(123)
        x = rng.standard_normal((8, 64)).astype(np.float32)
        small_pert = x + rng.standard_normal((8, 64)).astype(np.float32) * 0.001
        large_pert = x + rng.standard_normal((8, 64)).astype(np.float32) * 100.0
        small = LocalTrajectoryDivergence().compute_from_trajectory_pair(x, small_pert)["trajectory_divergence"]
        large = LocalTrajectoryDivergence().compute_from_trajectory_pair(x, large_pert)["trajectory_divergence"]
        assert large > small


class TestIntrinsicDimension:
    def test_twonn(self, hidden_states_2d):
        proxy = IntrinsicDimension(method="twonn")
        result = proxy.compute(hidden_states_2d)
        assert "intrinsic_dimension" in result
        assert result["intrinsic_dimension"] > 0

    def test_mle(self, hidden_states_2d):
        proxy = IntrinsicDimension(method="mle")
        result = proxy.compute(hidden_states_2d)
        assert result["intrinsic_dimension"] > 0

    def test_low_dim_manifold(self, rng):
        # Points on a 2D plane embedded in 64D should have ID ≈ 2
        base = rng.standard_normal((50, 2)).astype(np.float32)
        proj = np.zeros((50, 64), dtype=np.float32)
        proj[:, :2] = base
        proxy = IntrinsicDimension(method="twonn")
        result = proxy.compute(proj)
        # Allow generous tolerance
        assert 1.0 <= result["intrinsic_dimension"] <= 10.0

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            IntrinsicDimension(method="bad")


class TestNeighborhoodDistortionRatio:
    def test_basic_output(self, hidden_states_2d):
        proxy = NeighborhoodDistortionRatio(k=5)
        result = proxy.compute(hidden_states_2d)
        assert "neighborhood_distortion" in result
        assert 0.0 <= result["neighborhood_distortion"] <= 1.0

    def test_between_layers(self, rng):
        a = rng.standard_normal((20, 32)).astype(np.float32)
        b = a + rng.standard_normal((20, 32)).astype(np.float32) * 0.01
        proxy = NeighborhoodDistortionRatio(k=5)
        result = proxy.compute_between_layers(a, b)
        # Small perturbation should give low distortion
        assert result["neighborhood_distortion"] < 0.5


class TestGeodesicDeviationProxy:
    def test_basic_output(self, hidden_states_2d):
        proxy = GeodesicDeviationProxy(n_samples=20, k=5)
        result = proxy.compute(hidden_states_2d)
        assert "geodesic_deviation" in result
        assert isinstance(result["geodesic_deviation"], float)


class TestRicciGraphCurvature:
    def test_basic_output(self, hidden_states_2d):
        proxy = RicciGraphCurvature(k=5)
        result = proxy.compute(hidden_states_2d)
        assert "ricci_graph_curvature" in result
        assert isinstance(result["ricci_graph_curvature"], float)


class TestPersistentHomologyFeatures:
    def test_basic_output(self, hidden_states_2d):
        proxy = PersistentHomologyFeatures(max_dim=1, max_points=20)
        result = proxy.compute(hidden_states_2d)
        assert "betti_0" in result
        assert "persistence_entropy_0" in result
        assert result["betti_0"] >= 0

    def test_too_few_samples(self):
        proxy = PersistentHomologyFeatures(max_dim=1)
        result = proxy.compute(np.array([[1.0, 2.0]]))
        assert result["betti_0"] == 0.0


class TestCurvatureAggregator:
    def test_from_config_defaults(self):
        agg = CurvatureAggregator(proxy_names=["trajectory_divergence", "intrinsic_dimension"])
        assert len(agg.proxies) == 2

    def test_compute_layer(self, hidden_states_2d):
        agg = CurvatureAggregator(proxy_names=["trajectory_divergence"])
        bundle = agg.compute_layer(hidden_states_2d, layer_idx=0)
        assert bundle.layer_idx == 0
        assert "trajectory_divergence" in bundle.metrics
        assert bundle.primary() >= 0.0

    def test_compute_all_layers(self, hidden_states_3d):
        agg = CurvatureAggregator(proxy_names=["trajectory_divergence"])
        bundles = agg.compute_all_layers(hidden_states_3d)
        assert len(bundles) == hidden_states_3d.shape[0]
        for i, b in enumerate(bundles):
            assert b.layer_idx == i
