"""
LRP (Layer-wise Relevance Propagation) Score Computation Module.
This module provides functionality to compute LRP relevance scores for model weights,
which can then be used by the LRP-Merge method for intelligent model merging.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LRPConfig:
    """Configuration for LRP computation."""
    model_path: str
    output_path: str
    sample_prompts: List[str]
    batch_size: int = 1
    max_length: int = 512
    lrp_rule: str = "epsilon"  # "epsilon", "gamma", or "alpha_beta"
    epsilon: float = 1e-9
    gamma: float = 0.25
    alpha: float = 1.0
    beta: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LRPComputer:
    """
    Computes Layer-wise Relevance Propagation scores for transformer models.
    """

    def __init__(self, config: LRPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks: List[Any] = []

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Determine dtype based on device
        has_cuda = torch.cuda.is_available() and self.config.device == "cuda"
        if has_cuda:
            torch_dtype = torch.float16
            device_map = self.config.device
        else:
            torch_dtype = torch.float32
            device_map = None  # device_map not recommended for CPU

        print(f"  Using device: {self.config.device}")
        print(f"  Using dtype: {torch_dtype}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"  Failed with default settings: {e}")
            print("  Trying with trust_remote_code=True...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Explicitly move to CPU if needed
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
        Compute relevance using LRP-epsilon rule.
        R_j = sum_k (z_jk / (sum_j z_jk + epsilon)) * R_k
        """
        epsilon = self.config.epsilon
        
        # Ensure activations and weights are on same device
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        # Compute forward pass contribution z = xW^T
        z = F.linear(activations, weights)
        
        # Add epsilon for numerical stability
        z_stable = z + epsilon * torch.sign(z)

        # Compute redistribution factor s = R_out / z_stable
        s = output_relevance / z_stable
        
        # Flatten batch and sequence: (B, S, F) -> (N, F)
        x_flat = activations.reshape(-1, activations.shape[-1])
        s_flat = s.reshape(-1, s.shape[-1])
        
        # Relevance for weights: |W_ij * x_i * s_j| summed over batch/seq
        weight_relevance = weights.abs() * (s_flat.abs().t() @ x_flat.abs())

        return weight_relevance

    def compute_relevance_gamma(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relevance using LRP-gamma rule.
        Adds positive contributions with a gamma factor.
        """
        gamma = self.config.gamma
        epsilon = self.config.epsilon
        
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        # Separate positive contributions
        weights_pos = torch.clamp(weights, min=0)
        w_gamma = weights + gamma * weights_pos

        # Forward pass with enhanced weights
        z = F.linear(activations, w_gamma)
        z = z + epsilon * torch.sign(z)

        # Redistribute relevance
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
        Compute relevance using LRP-alpha_beta rule.
        Separates positive and negative contributions.
        """
        alpha = self.config.alpha
        beta = self.config.beta
        epsilon = self.config.epsilon
        
        activations = activations.to(weights.device)
        output_relevance = output_relevance.to(weights.device)

        weights_pos = torch.clamp(weights, min=0)
        weights_neg = torch.clamp(weights, max=0)

        # Positive and negative forward passes
        z_pos = F.linear(activations, weights_pos)
        z_neg = F.linear(activations, weights_neg)

        z = alpha * z_pos + beta * z_neg
        z = z + epsilon * torch.sign(z)

        s = output_relevance / z
        
        x_flat = activations.reshape(-1, activations.shape[-1])
        s_flat = s.reshape(-1, s.shape[-1])
        
        weight_relevance = weights.abs() * (s_flat.abs().t() @ x_flat.abs())

        return weight_relevance

    def compute_gradcam_importance(
        self,
        input_ids: torch.Tensor,
        target_layer: str,
    ) -> torch.Tensor:
        """
        Compute importance using Grad-CAM style gradients.
        This is a practical alternative to full LRP.
        """
        self.model.zero_grad()

        # Enable gradients for input
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        logits = outputs.logits

        # Compute gradient of output w.r.t. embeddings
        target_token_idx = logits.shape[1] - 1
        target_logit = logits[0, target_token_idx, :].max()
        target_logit.backward()

        # Get gradients
        gradients = inputs_embeds.grad

        # Importance = gradient magnitude
        importance = torch.abs(gradients)

        return importance

    def compute_relevance_for_tensor(
        self,
        tensor_name: str,
        tensor: torch.Tensor,
        sample_activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute relevance scores for a specific tensor.

        Args:
            tensor_name: Name of the tensor (e.g., "model.layers.0.attn.q_proj.weight")
            tensor: The weight tensor
            sample_activations: Optional sample activations from a forward pass

        Returns:
            Relevance scores for the tensor
        """
        # If we have sample activations, use proper LRP rules
        if sample_activations is not None:
            # Create dummy output relevance (normally from backward pass)
            # Match the shape of the output activations
            output_relevance = torch.ones_like(
                (
                    F.linear(sample_activations, tensor)
                    if tensor.dim() == 2
                    else sample_activations
                ),
                device=tensor.device,
            )

            if self.config.lrp_rule == "epsilon":
                relevance = self.compute_relevance_epsilon(
                    sample_activations, tensor, output_relevance
                )
            elif self.config.lrp_rule == "gamma":
                relevance = self.compute_relevance_gamma(
                    sample_activations, tensor, output_relevance
                )
            elif self.config.lrp_rule == "alpha_beta":
                relevance = self.compute_relevance_alpha_beta(
                    sample_activations, tensor, output_relevance
                )
            else:
                raise ValueError(f"Unknown LRP rule: {self.config.lrp_rule}")
        else:
            # Fallback: use magnitude-based proxy
            relevance = torch.abs(tensor)

        return relevance

    def _register_hooks(self) -> None:
        """Register forward hooks to collect activations."""
        self.activations = {}
        self.hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                # Store detached tensors; move to CPU if memory is an issue
                self.activations[name] = (input[0].detach(), output.detach())
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove previously registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def compute_all_relevance_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute relevance scores for all model weights.

        This is the main entry point for computing LRP scores.
        """
        if self.model is None:
            self.load_model()

        print(f"Computing LRP scores using {self.config.lrp_rule} rule...")

        # Tokenize sample prompts
        if self.config.sample_prompts:
            inputs = self.tokenizer(
                self.config.sample_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)

            # Get sample activations via forward pass with hooks
            self._register_hooks()
            with torch.no_grad():
                self.model(**inputs, output_hidden_states=False)
            self._remove_hooks()
        else:
            # No samples provided, use magnitude fallback
            print("No sample prompts provided, using magnitude-based importance...")

        # Map module names to parameter names
        # Most parameters in transformer layers follow {module_name}.weight or {module_name}.bias
        parameter_to_module = {}
        for mod_name, _ in self.model.named_modules():
            parameter_to_module[f"{mod_name}.weight"] = mod_name
            parameter_to_module[f"{mod_name}.bias"] = mod_name

        # Compute relevance for each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            print(f"Processing {name}...")

            # Try to find captured activations for this parameter's module
            module_name = parameter_to_module.get(name)
            act_data = self.activations.get(module_name)
            
            # Pass input activations (the first element of act_data tuple)
            sample_act = act_data[0] if act_data is not None else None

            # Compute relevance for this parameter
            relevance = self.compute_relevance_for_tensor(
                name, param.data, sample_activations=sample_act
            )

            self.relevance_scores[name] = relevance.cpu()

        return self.relevance_scores

    def save_relevance_scores(self, output_format: str = "safetensors") -> None:
        """Save computed relevance scores to disk."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_format == "safetensors":
            try:
                from safetensors.torch import save_file

                save_file(self.relevance_scores, output_path / "lrp_scores.safetensors")
            except ImportError:
                print("safetensors not available, using torch.save")
                torch.save(self.relevance_scores, output_path / "lrp_scores.pt")
        else:
            torch.save(self.relevance_scores, output_path / "lrp_scores.pt")

        # Save metadata
        metadata = {
            "model_path": self.config.model_path,
            "lrp_rule": self.config.lrp_rule,
            "epsilon": self.config.epsilon,
            "gamma": self.config.gamma,
            "num_tensors": len(self.relevance_scores),
        }

        with open(output_path / "lrp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"LRP scores saved to {output_path}")


def compute_lrp_for_model(
    model_path: str,
    output_path: str,
    sample_prompts: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute LRP scores for a model.

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute LRP scores for a model")
    parser.add_argument("model_path", help="Path to the HuggingFace model")
    parser.add_argument("output_path", help="Where to save LRP scores")
    parser.add_argument(
        "--rule", default="epsilon", choices=["epsilon", "gamma", "alpha_beta"]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--prompts", nargs="+", help="Sample prompts for LRP computation"
    )

    args = parser.parse_args()

    compute_lrp_for_model(
        model_path=args.model_path,
        output_path=args.output_path,
        sample_prompts=args.prompts,
        lrp_rule=args.rule,
        device=args.device,
    )
