#!/usr/bin/env python3
"""
Fine-tune models for fake news detection.
This script fine-tunes models on fake news datasets that can then be merged.
Optimized for CPU-only training environments with small models.

Recommended small models for CPU training:
- gpt2 (124M params) - Fastest
- gpt2-medium (355M params) - Good balance
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params) - Requires more memory
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """Dataset for fake news classification."""

    def __init__(
        self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create prompts
        self.prompts = [
            self._create_prompt(text, label) for text, label in zip(texts, labels)
        ]

    def _create_prompt(self, text: str, label: str) -> str:
        """Create a classification prompt."""
        return f"""Classify the following news article as FAKE or REAL.

Article: {text}

Classification: {label}"""

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),  # For causal LM
        }


def load_dataset(dataset_path: str, max_samples: Optional[int] = None) -> tuple:
    """Load dataset from CSV."""
    logger.info(f"Loading dataset from {dataset_path}...")

    df = pd.read_csv(dataset_path)

    # Handle different column names
    text_column = None
    label_column = None

    for possible_text in ["text", "content", "statement", "article", "title"]:
        if possible_text in df.columns:
            text_column = possible_text
            break

    for possible_label in ["label", "category", "truth", "class"]:
        if possible_label in df.columns:
            label_column = possible_label
            break

    if text_column is None or label_column is None:
        logger.error(
            f"Could not find text/label columns. Available: {list(df.columns)}"
        )
        sys.exit(1)

    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    # Normalize labels
    labels = [str(l).upper().strip() for l in labels]
    labels = [
        "FAKE" if l in ["FAKE", "0", "FALSE", "FALSE", "F"] else "REAL" for l in labels
    ]

    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"  FAKE: {labels.count('FAKE')}")
    logger.info(f"  REAL: {labels.count('REAL')}")

    return texts, labels


def setup_lora(model, r: int = 8, alpha: int = 32, dropout: float = 0.1):
    """Setup LoRA for efficient fine-tuning."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model

    except ImportError:
        logger.error("PEFT not installed. Run: pip install peft")
        sys.exit(1)


def fine_tune_model(
    dataset_path: str,
    output_dir: str,
    model_name: str = "gpt2",  # Default to small model for CPU
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    use_lora: bool = True,
    lora_r: int = 16,
    max_samples: Optional[int] = None,
    use_8bit: bool = False,
    use_4bit: bool = False,
    max_length: int = 128,
):
    """Fine-tune model on fake news detection.

    Args:
        use_8bit: Use 8-bit quantization (saves memory, slower on CPU)
        use_4bit: Use 4-bit quantization (saves more memory, slower on CPU)
        max_length: Maximum sequence length (lower = less memory)
    """

    logger.info(f"Starting fine-tuning...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  LoRA: {use_lora}")
    logger.info(f"  8-bit: {use_8bit}")
    logger.info(f"  4-bit: {use_4bit}")

    # Load data
    texts, labels = load_dataset(dataset_path, max_samples)

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # CPU/GPU compatibility logic
    has_cuda = torch.cuda.is_available()
    logger.info(f"  CUDA available: {has_cuda}")

    # Model loading kwargs
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }

    # Handle dtype - CPU doesn't support float16 well
    if has_cuda:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        # CPU training - use float32 and explicit device
        model_kwargs["torch_dtype"] = torch.float32
        # device_map not recommended for CPU-only
        logger.info("Using CPU with float32 precision")

    # Quantization for memory saving
    if use_8bit and has_cuda:
        model_kwargs["load_in_8bit"] = True
        logger.info("Loading model in 8-bit mode")
    elif use_4bit and has_cuda:
        model_kwargs["load_in_4bit"] = True
        logger.info("Loading model in 4-bit mode")
    elif (use_8bit or use_4bit) and not has_cuda:
        logger.warning(
            "8-bit/4-bit quantization requires CUDA. Using full precision on CPU."
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying fallback with trust_remote_code=True...")
        model_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Move to CPU explicitly if not using device_map
    if not has_cuda:
        model = model.to("cpu")

    # Enable gradient checkpointing for memory efficiency (if supported)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA if requested
    if use_lora:
        logger.info(f"Applying LoRA (r={lora_r})...")
        model = setup_lora(model, r=lora_r)

    # Create dataset
    logger.info("Creating dataset...")
    dataset = FakeNewsDataset(texts, labels, tokenizer, max_length=max_length)

    # Training arguments - optimized for CPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8 if not has_cuda else 4,  # Higher for CPU
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        # FP16 only on GPU
        fp16=has_cuda and not (use_8bit or use_4bit),
        # BF16 not on CPU
        bf16=False,
        dataloader_pin_memory=has_cuda,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=not has_cuda,
        # Disable some optimizations that don't work well on CPU
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        disable_tqdm=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyLlama for fake news detection"
    )

    parser.add_argument("--dataset", required=True, help="Path to training dataset CSV")

    parser.add_argument(
        "--output", required=True, help="Output directory for fine-tuned model"
    )

    parser.add_argument(
        "--model",
        default="gpt2",
        help="Base model to fine-tune (default: gpt2 for CPU, ~124M params)",
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,  # Default to 1 for CPU stability
        help="Training batch size",
    )

    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    parser.add_argument(
        "--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")

    parser.add_argument(
        "--max-samples", type=int, help="Limit training samples (for testing)"
    )

    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization (requires CUDA, saves memory)",
    )

    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (requires CUDA, saves more memory)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128, lower = less memory)",
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run fine-tuning
    fine_tune_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        max_samples=args.max_samples,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        max_length=args.max_length,
    )

    print("\n" + "=" * 60)
    print("Fine-tuned model saved to:", args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
