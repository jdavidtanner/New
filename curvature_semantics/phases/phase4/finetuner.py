"""LoRA/QLoRA fine-tuning on missing-link examples via PEFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def finetune_on_missing_links(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    output_dir: Path,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
) -> Any:
    """Apply LoRA fine-tuning to reduce curvature around missing bridge concepts.

    Returns the fine-tuned model (or original if PEFT is unavailable).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        import torch
    except ImportError:
        logger.warning("peft/transformers not fully installed; skipping fine-tuning")
        return model

    logger.info("Applying LoRA fine-tuning: r=%d, alpha=%d, epochs=%d", lora_r, lora_alpha, num_epochs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Build tokenized dataset
    texts = [f"{ex['prompt']} {ex.get('answer', '')}" for ex in examples]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    dataset = SimpleDataset(encodings)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        report_to=[],
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    logger.info("LoRA fine-tuning complete. Saved to %s", output_dir)
    return peft_model
