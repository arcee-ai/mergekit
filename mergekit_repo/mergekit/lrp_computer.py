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
from typing import Dict, List, Optional, Tuple, Any

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
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class LRPComputer:
    """Computes Layer-wise Relevance Propagation scores for transformer models."""

    def __init__(self, config: LRPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks: List[Any] = []

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

    def compute_relevance_epsilon(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Epsilon rule: R_j = sum_k (z_jk / (sum_j z_jk + epsilon)) * R_k
        """
        epsilon = self.config.epsilon
        
        # Ensure tensor devices match
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        # z = xW^T
        z = F.linear(activations, weights)
        z_stable = z + epsilon * torch.sign(z)
        
        s = output_relevance / z_stable
        
        x_flat = activations.reshape(-1, activations.shape[-1])
        s_flat = s.reshape(-1, s.shape[-1])
        
        # Weight relevance: |W_ij * x_i * s_j|
        weight_relevance = weights.abs() * (s_flat.abs().t() @ x_flat.abs())
        
        return weight_relevance

    def compute_relevance_gamma(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gamma rule: Enhances positive contributions by gamma factor.
        """
        gamma = self.config.gamma
        epsilon = self.config.epsilon
        
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        weights_pos = torch.clamp(weights, min=0)
        w_gamma = weights + gamma * weights_pos

        z = F.linear(activations, w_gamma)
        z = z + epsilon * torch.sign(z)

        s = output_relevance / z
        
        x_flat = activations.reshape(-1, activations.shape[-1])
        s_flat = s.reshape(-1, s.shape[-1])
        
        weight_relevance = weights.abs() * (s_flat.abs().t() @ x_flat.abs())

        return weight_relevance

    def compute_relevance_alpha_beta(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alpha-beta rule: Separates positive and negative contributions.
        """
        alpha = self.config.alpha
        beta = self.config.beta
        epsilon = self.config.epsilon
        
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        weights_pos = torch.clamp(weights, min=0)
        weights_neg = torch.clamp(weights, max=0)

        z_pos = F.linear(activations, weights_pos)
        z_neg = F.linear(activations, weights_neg)

        z = alpha * z_pos + beta * z_neg
        z = z + epsilon * torch.sign(z)

        s = output_relevance / z
        
        x_flat = activations.reshape(-1, activations.shape[-1])
        s_flat = s.reshape(-1, s.shape[-1])
        
        weight_relevance = weights.abs() * (s_flat.abs().t() @ x_flat.abs())

        return weight_relevance

    def compute_relevance_for_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        sample_activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute relevance for a specific tensor."""
        if sample_activations is not None:
            # Create dummy output relevance
            output_relevance = torch.ones_like(
                F.linear(sample_activations, tensor) if tensor.dim() == 2 else sample_activations,
                device=tensor.device
            )

            rule = self.config.lrp_rule
            if rule == "epsilon":
                return self.compute_relevance_epsilon(sample_activations, tensor, output_relevance)
            elif rule == "gamma":
                return self.compute_relevance_gamma(sample_activations, tensor, output_relevance)
            elif rule == "alpha_beta":
                return self.compute_relevance_alpha_beta(sample_activations, tensor, output_relevance)
        
        return torch.abs(tensor)

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

    def _register_hooks(self) -> None:
        """Register forward hooks to collect activations."""
        self.activations = {}
        self.hooks = []
        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = (input[0].detach(), output.detach())
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def _remove_hooks(self) -> None:
        """Remove previously registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

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

            # Collect activations via forward pass
            self._register_hooks()
            with torch.no_grad():
                self.model(**inputs, output_hidden_states=False)
            self._remove_hooks()
        else:
            logger.info("No sample prompts provided, using magnitude-based fallback...")

        # Map parameters to modules
        parameter_to_module = {}
        for mod_name, _ in self.model.named_modules():
            parameter_to_module[f"{mod_name}.weight"] = mod_name
            parameter_to_module[f"{mod_name}.bias"] = mod_name

        # Compute relevance for each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            module_name = parameter_to_module.get(name)
            act_data = self.activations.get(module_name)
            sample_act = act_data[0] if act_data else None

            self.relevance_scores[name] = self.compute_relevance_for_tensor(
                name, param.data, sample_activations=sample_act
            ).cpu()

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
        "--epsilon",
        type=float,
        default=1e-9,
        help="Epsilon value for epsilon rule"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="Gamma value for gamma rule"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha value for alpha_beta rule"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Beta value for alpha_beta rule"
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
        epsilon=args.epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
