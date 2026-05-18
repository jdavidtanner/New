"""Extract layerwise hidden states via forward hooks; supports MoE routing capture."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class HiddenStateBundle:
    """Container for per-layer hidden states extracted from a forward pass."""

    layer_indices: list[int]
    # shape: (num_layers, batch, seq_len, hidden_dim)
    hidden_states: list[torch.Tensor] = field(default_factory=list)
    # optional: last-token representation per layer
    last_token: list[torch.Tensor] | None = None
    # logits from final LM head (batch, seq_len, vocab)
    logits: torch.Tensor | None = None
    # MoE routing weights if available
    routing_weights: dict[int, torch.Tensor] = field(default_factory=dict)

    def last_token_array(self) -> np.ndarray:
        """Return (num_layers, batch, hidden_dim) float32 array of last-token states."""
        tensors = self.last_token or [h[:, -1, :] for h in self.hidden_states]
        return torch.stack(tensors).float().cpu().numpy()

    def mean_pool_array(self) -> np.ndarray:
        """Return (num_layers, batch, hidden_dim) mean-pooled over sequence."""
        pooled = [h.mean(dim=1) for h in self.hidden_states]
        return torch.stack(pooled).float().cpu().numpy()


class HiddenStateExtractor:
    """Registers forward hooks on transformer layers and collects hidden states."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int] | None = None,
        extract_logits: bool = True,
        capture_routing: bool = False,
    ):
        self.model = model
        self.extract_logits = extract_logits
        self.capture_routing = capture_routing
        self._hooks: list[Any] = []
        self._cache: dict[int, list[torch.Tensor]] = {}
        self._routing_cache: dict[int, torch.Tensor] = {}
        self._logits_cache: torch.Tensor | None = None

        self._layers = self._discover_layers()
        n = len(self._layers)
        if layer_indices is None:
            self.layer_indices = list(range(n))
        else:
            # Support negative indices
            self.layer_indices = [i % n for i in layer_indices]

    def _discover_layers(self) -> nn.ModuleList | list[nn.Module]:
        """Find the main transformer decoder layers."""
        for attr in ("model.layers", "transformer.h", "model.decoder.layers", "layers"):
            parts = attr.split(".")
            obj = self.model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        logger.warning("Could not auto-discover transformer layers; using empty list")
        return []

    @contextlib.contextmanager
    def capture(self):
        """Context manager that registers hooks and clears cache on entry/exit."""
        self._cache.clear()
        self._routing_cache.clear()
        self._logits_cache = None
        try:
            self._register_hooks()
            yield self
        finally:
            self._remove_hooks()

    def _register_hooks(self) -> None:
        for idx in self.layer_indices:
            if idx >= len(self._layers):
                continue
            layer = self._layers[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

        if self.extract_logits:
            handle = self.model.register_forward_hook(self._logits_hook)
            self._hooks.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            # output can be a tuple (hidden, ...) or just a tensor
            if isinstance(output, (tuple, list)):
                hs = output[0]
                # MoE routing weights are often in output[1] or output[2]
                if self.capture_routing and len(output) > 1:
                    for item in output[1:]:
                        if isinstance(item, torch.Tensor) and item.ndim == 2:
                            self._routing_cache[idx] = item.detach().cpu()
                            break
            else:
                hs = output
            self._cache.setdefault(idx, []).append(hs.detach().cpu())
        return hook

    def _logits_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self._logits_cache = output.detach().cpu()
        elif hasattr(output, "logits"):
            self._logits_cache = output.logits.detach().cpu()

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def build_bundle(self) -> HiddenStateBundle:
        layers = sorted(self._cache.keys())
        hidden_states = []
        for idx in layers:
            # Concatenate across batch chunks if needed
            chunks = self._cache[idx]
            hs = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            hidden_states.append(hs)

        bundle = HiddenStateBundle(
            layer_indices=layers,
            hidden_states=hidden_states,
            logits=self._logits_cache,
            routing_weights=dict(self._routing_cache),
        )
        bundle.last_token = [h[:, -1, :] for h in hidden_states]
        return bundle

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> HiddenStateBundle:
        with self.capture():
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.model(input_ids=input_ids)
        return self.build_bundle()
