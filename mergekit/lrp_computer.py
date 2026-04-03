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
            # Gamma rule: enhance positive contributions with (1 + gamma) factor
            gamma = self.config.gamma
            pos_output = F.relu(layer_output)
            neg_output = F.relu(-layer_output)
            pos_weight = (
                (1 + gamma) * pos_output / (pos_output.sum(dim=-1, keepdim=True) + 1e-9)
            )
            neg_weight = neg_output / (neg_output.sum(dim=-1, keepdim=True) + 1e-9)
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

            if self.config.lrp_rule in ("epsilon", "gamma", "alpha_beta"):
                self.relevance_scores = self._compute_lrp_importance(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )
            else:
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

    def _compute_lrp_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance using the configured LRP rule by propagating relevance
        backward through each layer's activations.

        Relevance starts at the final token logits, is projected back through the
        lm_head weight to hidden_size, then propagated through each layer activation
        using the configured LRP rule.
        """
        importance_scores = {}

        # Collect activations via forward hooks
        activations: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = (
                    out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
                )

            return hook

        handles = []
        for name, module in self.model.named_modules():
            handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        for h in handles:
            h.remove()

        # Initial relevance: (vocab_size,) from final token logits
        logits = outputs.logits  # (batch, seq, vocab)
        relevance_vocab = logits[:, -1, :].abs().mean(dim=0)  # (vocab_size,)

        # Project relevance from vocab_size -> hidden_size via lm_head weight.
        # lm_head.weight shape is (vocab_size, hidden_size), so W^T @ R_vocab
        # gives a (hidden_size,) relevance signal that can be broadcast over layers.
        hidden_relevance = None
        for name, param in self.model.named_parameters():
            if "lm_head" in name and param.dim() == 2:
                # param: (vocab_size, hidden_size)
                if param.shape[0] == relevance_vocab.shape[0]:
                    hidden_relevance = (
                        (relevance_vocab.unsqueeze(0) @ param.float()).squeeze(0).abs()
                    )  # (hidden_size,)
                    break

        # Propagate relevance through each named parameter using the chosen rule
        for name, param in self.model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            act = activations.get(layer_name)
            if act is None:
                importance_scores[name] = param.data.abs().cpu()
                continue

            # Flatten act to (tokens, features)
            act_flat = act.reshape(-1, act.shape[-1]) if act.dim() > 1 else act
            feat_dim = act_flat.shape[-1]

            # Build a relevance signal matching act_flat's feature dimension
            if hidden_relevance is not None and hidden_relevance.numel() == feat_dim:
                # Broadcast (hidden_size,) across all tokens
                rel_signal = hidden_relevance.expand_as(act_flat)
            else:
                # Fall back to uniform signal scaled by mean activation magnitude
                rel_signal = act_flat.abs().mean() * torch.ones_like(act_flat)

            rel = self.compute_layer_relevance(
                layer_input=act_flat,
                layer_output=act_flat,
                relevance_output=rel_signal,
                layer_name=layer_name,
            )

            # Reduce (tokens, features) -> param shape safely
            reduced = rel.abs().mean(dim=0)  # (features,)
            if reduced.numel() == param.numel():
                importance_scores[name] = reduced.reshape(param.shape).cpu()
            else:
                importance_scores[name] = param.data.abs().cpu()

        return importance_scores

    def save_relevance_scores(self) -> None:
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
