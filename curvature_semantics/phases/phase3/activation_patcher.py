"""Activation patching: inject calibrated noise at a specific transformer layer.

The causal test:
  - Same prompt, same model weights
  - At layer L, replace h_L with h_L + alpha * ||h_L||_mean * eps, eps~N(0,I)
  - Continue autoregressive generation from the perturbed representation
  - If curvature (alpha) causes lower semantic completeness, this is causal evidence
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _discover_layers(model: nn.Module) -> list[nn.Module]:
    """Same discovery logic as HiddenStateExtractor."""
    for attr in ("model.layers", "transformer.h", "model.decoder.layers", "layers"):
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    return []


class ActivationPatcher:
    """Patches a single transformer layer's output during the prefill forward pass."""

    def __init__(self, model: nn.Module, rng_seed: int = 42):
        self.model = model
        self.layers = _discover_layers(model)
        self._rng = np.random.default_rng(rng_seed)
        self._handle: Any = None
        self._captured: np.ndarray | None = None

    @contextlib.contextmanager
    def patch(self, layer_idx: int, alpha: float):
        """Context manager that injects noise at layer_idx during prefill.

        alpha: noise scale as a fraction of the mean activation L2 norm.
               alpha=0 → no-op (baseline), alpha=1 → noise magnitude ≈ activation norm.
        Only applies during the prefill step (seq_len > 1); autoregressive
        decode steps are left untouched so the prompt representation is what
        we control.
        """
        self._captured = None
        seed = int(self._rng.integers(0, 2**31))

        def _hook(module: nn.Module, inp: tuple, output: Any) -> Any:
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == 1:
                # Autoregressive step — don't patch
                return output
            # Prefill: inject noise proportional to per-position activation norm
            if alpha > 0.0:
                gen = torch.Generator(device=hs.device).manual_seed(seed)
                noise = torch.randn(hs.shape, generator=gen, dtype=hs.dtype, device=hs.device)
                scale = hs.norm(dim=-1, keepdim=True).mean() * alpha
                hs = hs + noise * scale
            # Capture last-token representation at this layer
            self._captured = hs[:, -1, :].detach().float().cpu().numpy()
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        if layer_idx >= len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range (model has {len(self.layers)} layers)")

        handle = self.layers[layer_idx].register_forward_hook(_hook)
        try:
            yield self
        finally:
            handle.remove()

    def captured_activation(self) -> np.ndarray | None:
        """Return (batch, hidden_dim) last-token activation captured at the patched layer."""
        return self._captured

    def n_layers(self) -> int:
        return len(self.layers)
