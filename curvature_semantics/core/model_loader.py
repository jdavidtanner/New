"""HuggingFace model + tokenizer loading with device placement and quantization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class _MockTokenizer:
    """Tiny deterministic tokenizer for proof-of-concept runs with no downloads."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation, padding
        words = text.split() or [text]
        ids = [2 + (sum(ord(ch) for ch in word) % 125) for word in words[:max_length]]
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [tok for tok in token_ids if tok not in (self.pad_token_id, self.eos_token_id)]
        return " ".join(f"tok{int(tok)}" for tok in token_ids) or "mock response"


class _MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.tanh(self.proj(hidden)),)


class _MockCausalLM(nn.Module):
    """Small transformer-like causal LM that supports hooks and generation."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, num_layers: int = 4):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_MockLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(num_hidden_layers=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.embed(input_ids % self.embed.num_embeddings)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        new_len = max(1, min(max_new_tokens, 8))
        generated = input_ids.clone()
        for _ in range(new_len):
            logits = self(generated).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def _is_mock_model(model_id: str) -> bool:
    return model_id in {"mock://tiny-causal-lm", "mock-tiny", "mock"}


def _load_mock_model_and_tokenizer(device: str) -> tuple[Any, Any]:
    model = _MockCausalLM().to(device)
    model.eval()
    return model, _MockTokenizer()


def resolve_device(device: str = "cuda") -> str:
    """Return an available torch device, falling back to CPU when necessary."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return "cpu"
    return device


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer.

    Returns (model, tokenizer).
    """
    device = resolve_device(device)
    if _is_mock_model(model_id):
        logger.info("Loading mock model for proof-of-concept run: %s", model_id)
        return _load_mock_model_and_tokenizer(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPE_MAP.get(dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16) if not (load_in_4bit or load_in_8bit) else None

    logger.info("Loading model: %s  device=%s  dtype=%s", model_id, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map=device if (load_in_4bit or load_in_8bit) else None,
        trust_remote_code=True,
        **kwargs,
    )
    if not (load_in_4bit or load_in_8bit):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def get_num_layers(model: Any) -> int:
    """Return the number of transformer decoder layers."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        n = getattr(getattr(model, "config", None), attr, None)
        if n is not None:
            return int(n)
    # Fallback: count named children that look like layers
    count = sum(1 for name, _ in model.named_modules() if "layers." in name and name.count(".") == 2)
    return max(count, 1)


def model_param_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())
