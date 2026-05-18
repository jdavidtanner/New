"""Central dispatch for all curvature proxies; returns a unified CurvatureBundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.curvature.base import CurvatureProxy
from curvature_semantics.curvature.geodesic_deviation import GeodesicDeviationProxy
from curvature_semantics.curvature.intrinsic_dimension import IntrinsicDimension
from curvature_semantics.curvature.neighborhood_distortion import NeighborhoodDistortionRatio
from curvature_semantics.curvature.persistent_homology import PersistentHomologyFeatures
from curvature_semantics.curvature.ricci_graph_curvature import RicciGraphCurvature
from curvature_semantics.curvature.trajectory_divergence import LocalTrajectoryDivergence

logger = get_logger(__name__)

_PROXY_REGISTRY: dict[str, type[CurvatureProxy]] = {
    "trajectory_divergence": LocalTrajectoryDivergence,
    "intrinsic_dimension": IntrinsicDimension,
    "neighborhood_distortion": NeighborhoodDistortionRatio,
    "geodesic_deviation": GeodesicDeviationProxy,
    "ricci_graph_curvature": RicciGraphCurvature,
    "persistent_homology": PersistentHomologyFeatures,
}


@dataclass
class CurvatureBundle:
    """All curvature proxy values for a single (layer, batch) pair."""
    layer_idx: int
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> float:
        """Return the primary curvature signal (trajectory divergence)."""
        return self.metrics.get("trajectory_divergence", 0.0)

    def to_flat_dict(self) -> dict[str, Any]:
        d = {"layer_idx": self.layer_idx, "n_samples": self.n_samples}
        d.update(self.metrics)
        d.update(self.metadata)
        return d


class CurvatureAggregator:
    """Runs all enabled curvature proxies and returns CurvatureBundle per layer."""

    def __init__(self, proxy_names: list[str] | None = None, proxy_params: dict[str, Any] | None = None):
        proxy_names = proxy_names or list(_PROXY_REGISTRY.keys())
        proxy_params = proxy_params or {}
        self.proxies: list[CurvatureProxy] = []
        for name in proxy_names:
            cls = _PROXY_REGISTRY.get(name)
            if cls is None:
                logger.warning("Unknown curvature proxy: %s — skipping", name)
                continue
            params = proxy_params.get(name, {})
            self.proxies.append(cls(**params))

    @classmethod
    def from_config(cls, cfg: Any) -> CurvatureAggregator:
        proxy_names = cfg.raw.get("curvature_proxies", list(_PROXY_REGISTRY.keys()))
        proxy_params = cfg.raw.get("curvature_params", {})
        return cls(proxy_names=proxy_names, proxy_params=proxy_params)

    def compute_layer(self, hidden_states: np.ndarray, layer_idx: int = 0) -> CurvatureBundle:
        """Compute all proxies for a single layer's hidden states.

        Args:
            hidden_states: (n_samples, hidden_dim) float array
            layer_idx: which transformer layer this came from
        """
        hidden_states = np.asarray(hidden_states)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states[np.newaxis, :]
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        bundle = CurvatureBundle(layer_idx=layer_idx, n_samples=hidden_states.shape[0])
        for proxy in self.proxies:
            try:
                result = proxy.compute(hidden_states)
                bundle.metrics.update(result)
            except Exception as exc:
                logger.warning("Proxy %s failed at layer %d: %s", proxy.name, layer_idx, exc)
        return bundle

    def compute_all_layers(
        self,
        layerwise_states: np.ndarray,
    ) -> list[CurvatureBundle]:
        """Compute curvature bundles for each layer.

        Args:
            layerwise_states: (n_layers, n_samples, hidden_dim)
        """
        bundles = []
        for layer_idx in range(layerwise_states.shape[0]):
            hs = layerwise_states[layer_idx]
            bundle = self.compute_layer(hs, layer_idx=layer_idx)
            bundles.append(bundle)
        return bundles
