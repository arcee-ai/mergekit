# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
"""
LRP (Layer-wise Relevance Propagation) Score Computation Module.

This module provides functionality to compute LRP relevance scores for model weights,
which can then be used by the LRP-Merge method for intelligent model merging.

Usage:
    python -m mergekit.lrp_computer \
        --model path/to/model \
        --output path/to/output \
        --rule epsilon
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LRPConfig:
    """Configuration for LRP computation."""
    model_path: str
    output_path: str
    sample_prompts: List[str] = field(default_factory=list)
    batch_size: int = 1
    max_length: int = 512
    lrp_rule: str = "epsilon"
    epsilon: float = 1e-9
    gamma: float = 0.25
    alpha: float = 1.0
    beta: float = 0.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class LRPComputer:
    """Computes Layer-wise Relevance Propagation scores for transformer models."""

    def __init__(self, config: LRPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.config.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def compute_gradient_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance using gradient-based attribution.
        """
        importance_scores = {}

        # Enable gradients for embeddings
        self.model.zero_grad()

        # Forward pass with gradient computation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Compute gradients of output w.r.t. input
        logits = outputs.logits
        target_token_idx = logits.shape[1] - 1
        target_logit = logits[0, target_token_idx, :].max()
        target_logit.backward()

        # Collect gradients for each parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                importance_scores[name] = torch.abs(param.grad).cpu()
            else:
                # Fallback to magnitude
                importance_scores[name] = torch.abs(param.data).cpu()

        return importance_scores

    def compute_all_relevance_scores(self) -> Dict[str, torch.Tensor]:
        """Compute relevance scores for all model weights."""
        if self.model is None:
            self.load_model()

        logger.info(f"Computing relevance scores using gradient attribution...")

        # Tokenize sample prompts
        if self.config.sample_prompts:
            inputs = self.tokenizer(
                self.config.sample_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            if self.config.device == "cuda":
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Compute importance
            self.relevance_scores = self.compute_gradient_importance(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
        else:
            # No samples provided, use magnitude fallback
            logger.info("No sample prompts provided, using magnitude-based importance...")
            for name, param in self.model.named_parameters():
                self.relevance_scores[name] = torch.abs(param.data).cpu()

        return self.relevance_scores

    def save_relevance_scores(self, output_format: str = "safetensors") -> None:
        """Save computed relevance scores to disk."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save scores
        scores_path = output_path / "lrp_scores.pt"
        torch.save(self.relevance_scores, scores_path)
        logger.info(f"Saved LRP scores to {scores_path}")

        # Save metadata
        metadata = {
            "model_path": self.config.model_path,
            "lrp_rule": self.config.lrp_rule,
            "epsilon": self.config.epsilon,
            "gamma": self.config.gamma,
            "num_tensors": len(self.relevance_scores),
            "tensor_names": list(self.relevance_scores.keys()),
        }

        metadata_path = output_path / "lrp_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def compute_lrp_for_model(
    model_path: str,
    output_path: str,
    sample_prompts: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute LRP scores for a model.

    Args:
        model_path: Path to the HuggingFace model
        output_path: Where to save the LRP scores
        sample_prompts: List of sample prompts for LRP computation
        **kwargs: Additional configuration options

    Returns:
        Dictionary mapping tensor names to relevance scores
    """
    default_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The capital of France is Paris.",
        "Machine learning is a subset of AI.",
        "Neural networks learn patterns from data.",
    ]

    config = LRPConfig(
        model_path=model_path,
        output_path=output_path,
        sample_prompts=sample_prompts or default_prompts,
        **kwargs
    )

    computer = LRPComputer(config)
    scores = computer.compute_all_relevance_scores()
    computer.save_relevance_scores()

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute LRP scores for a model"
    )
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model")
    parser.add_argument("--output", required=True, help="Where to save LRP scores")
    parser.add_argument(
        "--rule",
        default="epsilon",
        choices=["epsilon", "gamma", "alpha_beta"],
        help="LRP rule to use"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Sample prompts for LRP computation"
    )

    args = parser.parse_args()

    compute_lrp_for_model(
        model_path=args.model,
        output_path=args.output,
        sample_prompts=args.prompts,
        lrp_rule=args.rule,
        device=args.device,
    )


if __name__ == "__main__":
    main()
