"""HuggingFace model + tokenizer loading with device placement and quantization."""

from __future__ import annotations

from typing import Any

import torch

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = resolve_device(device)
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
