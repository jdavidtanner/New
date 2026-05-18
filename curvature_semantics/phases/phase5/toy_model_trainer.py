"""Trains small GPT-2-like models on ontology-derived corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.utils.checkpoint_utils import checkpoint_path

logger = get_logger(__name__)


def build_corpus(
    sentences: list[str],
    train_size: int = 5000,
    val_size: int = 500,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Expand and split a sentence list into train/val corpora."""
    import random
    rng = random.Random(seed)
    # Augment by repeating with slight variations if corpus is small
    corpus = list(sentences)
    while len(corpus) < train_size + val_size:
        corpus.extend(rng.sample(sentences, min(len(sentences), 100)))
    rng.shuffle(corpus)
    return corpus[:train_size], corpus[train_size: train_size + val_size]


def train_toy_model(
    train_sentences: list[str],
    val_sentences: list[str],
    output_dir: Path,
    hidden_size: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    max_position_embeddings: int = 256,
    vocab_size: int = 2000,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    checkpoint_every: int = 5,
) -> Any:
    """Train a small GPT-2 config model from scratch on the ontology corpus.

    Returns the trained model (or None if torch/transformers unavailable).
    """
    try:
        import torch
        from transformers import (
            GPT2Config, GPT2LMHeadModel,
            PreTrainedTokenizerFast,
            get_linear_schedule_with_warmup,
        )
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        logger.warning("transformers/tokenizers not installed; skipping toy model training")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a tiny BPE tokenizer on the corpus
    logger.info("Training BPE tokenizer on ontology corpus")
    tokenizer_obj = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer_obj.train_from_iterator(train_sentences + val_sentences, trainer=trainer)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_obj.save(str(tokenizer_path))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.add_special_tokens({
        "unk_token": "[UNK]", "pad_token": "[PAD]",
        "bos_token": "[BOS]", "eos_token": "[EOS]",
    })

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=max_position_embeddings,
        n_ctx=max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    logger.info("Toy model: %dM params", sum(p.numel() for p in model.parameters()) // 1_000_000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    def encode_batch(texts: list[str]) -> torch.Tensor:
        enc = tokenizer(texts, truncation=True, padding=True, max_length=max_position_embeddings, return_tensors="pt")
        return enc["input_ids"].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = num_epochs * (len(train_sentences) // batch_size + 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(num_epochs):
        model.train()
        import random
        random.shuffle(train_sentences)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_sentences), batch_size):
            batch_texts = train_sentences[i: i + batch_size]
            input_ids = encode_batch(batch_texts)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — train loss=%.4f", epoch + 1, num_epochs, avg_loss)

        if (epoch + 1) % checkpoint_every == 0:
            ckpt = checkpoint_path(output_dir / "checkpoints", epoch + 1)
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "loss": avg_loss}, ckpt)

    model.save_pretrained(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    logger.info("Toy model training complete. Saved to %s", output_dir / "final_model")
    return model
