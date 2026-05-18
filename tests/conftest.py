"""Shared fixtures: tiny mock model, sample hidden states, toy config."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def hidden_states_2d(rng):
    """(30, 64) float32 array — 30 samples in 64-dim space."""
    return rng.standard_normal((30, 64)).astype(np.float32)


@pytest.fixture
def hidden_states_3d(rng):
    """(8, 30, 64) float32 array — 8 layers, 30 samples, 64-dim."""
    return rng.standard_normal((8, 30, 64)).astype(np.float32)


@pytest.fixture
def tiny_model():
    """Minimal 2-layer transformer-like module for hook testing."""

    class TinyLayer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.linear = nn.Linear(d, d)

        def forward(self, x):
            return (self.linear(x),)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(64) for _ in range(4)])
            self.lm_head = nn.Linear(64, 100)

        def forward(self, input_ids, attention_mask=None):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], 64)
            for layer in self.layers:
                x = layer(x)[0]
            logits = self.lm_head(x)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)

        @property
        def config(self):
            from types import SimpleNamespace
            return SimpleNamespace(num_hidden_layers=4)

    return TinyModel()


@pytest.fixture
def sample_nli_pair():
    return (
        "The cat sat on the mat.",
        "A feline rested on a rug.",
    )
