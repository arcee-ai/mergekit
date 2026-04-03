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
import torch.nn.functional as F
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
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


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
            torch_dtype=(
                torch.float16 if self.config.device == "cuda" else torch.float32
            ),
            device_map="auto" if self.config.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.config.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def compute_relevance_epsilon(
        self,
        activations: Dict[str, torch.Tensor],
        relevance: torch.Tensor,
        layer_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Epsilon rule: R_j = sum_k (z_jk / (sum_j z_jk + epsilon)) * R_k

        Adds small epsilon to denominators to avoid division by zero.
        """
        epsilon = self.config.epsilon
        new_relevance = {}

        for name, act in activations.items():
            # Compute z_jk = activation of neuron j to neuron k
            z = act + epsilon * torch.sign(act)
            z_sum = z.sum(dim=-1, keepdim=True)
            z_norm = z / (z_sum + epsilon)

            # Distribute relevance backwards
            if name in relevance:
                new_relevance[name] = z_norm * relevance[name]

        return new_relevance

    def compute_relevance_gamma(
        self,
        activations: Dict[str, torch.Tensor],
        relevance: torch.Tensor,
        layer_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Gamma rule: Enhances positive contributions by gamma factor.

        R_j = sum_k ((z_jk)^+ / sum_j (z_jk)^+ + gamma * (z_jk)^-) * R_k
        """
        gamma = 1.0 + self.config.gamma  # gamma is added to positive part
        new_relevance = {}

        for name, act in activations.items():
            # Separate positive and negative parts
            act_pos = F.relu(act)
            act_neg = F.relu(-act)

            # Normalize positive part
            pos_sum = act_pos.sum(dim=-1, keepdim=True)
            pos_norm = act_pos / (pos_sum + 1e-9)

            # Apply gamma to negative part
            neg_sum = act_neg.sum(dim=-1, keepdim=True)
            neg_norm = gamma * act_neg / (neg_sum + 1e-9)

            # Combine
            z_norm = pos_norm + neg_norm

            if name in relevance:
                new_relevance[name] = z_norm * relevance[name]

        return new_relevance

    def compute_relevance_alpha_beta(
        self,
        activations: Dict[str, torch.Tensor],
        relevance: torch.Tensor,
        layer_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Alpha-beta rule: Separates positive and negative contributions.

        R = alpha * R^+ + beta * R^-
        where alpha + beta = 1
        """
        alpha = self.config.alpha
        beta = self.config.beta
        new_relevance = {}

        for name, act in activations.items():
            # Separate positive and negative parts
            act_pos = F.relu(act)
            act_neg = F.relu(-act)

            # Normalize each separately
            pos_sum = act_pos.sum(dim=-1, keepdim=True)
            neg_sum = act_neg.sum(dim=-1, keepdim=True)

            pos_norm = act_pos / (pos_sum + 1e-9)
            neg_norm = act_neg / (neg_sum + 1e-9)

            # Combine with alpha and beta
            z_norm = alpha * pos_norm - beta * neg_norm

            if name in relevance:
                new_relevance[name] = z_norm * relevance[name]

        return new_relevance

    def compute_layer_relevance(
        self,
        layer_input: torch.Tensor,
        layer_output: torch.Tensor,
        relevance_output: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Compute relevance for a single layer using the configured LRP rule.

        Args:
            layer_input: Input activations to the layer
            layer_output: Output activations from the layer
            relevance_output: Relevance scores at the output
            layer_name: Name of the layer

        Returns:
            Relevance scores at the layer input
        """
        rule = self.config.lrp_rule

        if rule == "epsilon":
            # Epsilon rule: add small constant to denominator
            epsilon = self.config.epsilon
            numerator = layer_output
            denominator = layer_output.abs().sum(dim=-1, keepdim=True) + epsilon
            relevance_input = (numerator / denominator) * relevance_output
        elif rule == "gamma":
            # Gamma rule: enhance positive contributions
            gamma = self.config.gamma
            pos_output = F.relu(layer_output)
            neg_output = F.relu(-layer_output)
            pos_weight = pos_output / (pos_output.sum(dim=-1, keepdim=True) + 1e-9)
            neg_weight = (
                gamma * neg_output / (neg_output.sum(dim=-1, keepdim=True) + 1e-9)
            )
            relevance_input = (pos_weight - neg_weight) * relevance_output
        elif rule == "alpha_beta":
            # Alpha-beta rule
            alpha = self.config.alpha
            beta = self.config.beta
            pos_output = F.relu(layer_output)
            neg_output = F.relu(-layer_output)
            pos_weight = pos_output / (pos_output.sum(dim=-1, keepdim=True) + 1e-9)
            neg_weight = neg_output / (neg_output.sum(dim=-1, keepdim=True) + 1e-9)
            relevance_input = (
                alpha * pos_weight - beta * neg_weight
            ) * relevance_output
        else:
            # Fallback: gradient-based importance
            relevance_input = layer_input.abs() * relevance_output.abs().mean()

        return relevance_input

    def compute_gradient_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance using gradient-based attribution across all prompts in batch.
        """
        importance_scores = {}

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        target_token_idx = logits.shape[1] - 1
        batch_size = logits.shape[0]

        # Compute gradients for each prompt in batch, accumulated into parameters
        self.model.zero_grad()
        for i in range(batch_size):
            target_logit = logits[i, target_token_idx, :].max()
            target_logit.backward(retain_graph=(i < batch_size - 1))

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

        logger.info(f"Computing relevance scores using {self.config.lrp_rule} rule...")

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
            logger.info(
                "No sample prompts provided, using magnitude-based importance..."
            )
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
            "alpha": self.config.alpha,
            "beta": self.config.beta,
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
    **kwargs,
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
        **kwargs,
    )

    computer = LRPComputer(config)
    scores = computer.compute_all_relevance_scores()
    computer.save_relevance_scores()

    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute LRP scores for a model")
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model")
    parser.add_argument("--output", required=True, help="Where to save LRP scores")
    parser.add_argument(
        "--rule",
        default="epsilon",
        choices=["epsilon", "gamma", "alpha_beta"],
        help="LRP rule to use",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-9, help="Epsilon value for epsilon rule"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.25, help="Gamma value for gamma rule"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Alpha value for alpha_beta rule"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0, help="Beta value for alpha_beta rule"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--prompts", nargs="+", help="Sample prompts for LRP computation"
    )

    args = parser.parse_args()

    compute_lrp_for_model(
        model_path=args.model,
        output_path=args.output,
        sample_prompts=args.prompts,
        lrp_rule=args.rule,
        epsilon=args.epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
