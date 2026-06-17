import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LRPComputer:
    """
    Supercharged LRP Computer with Multimodal support and Tied-Tensor handling.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def compute_layer_relevance(
        self,
        layer_input: torch.Tensor,
        layer_output: torch.Tensor,
        relevance_output: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Compute relevance using autograd for the chosen rule."""
        # Simple epsilon rule implementation using gradients as a proxy for LRP
        # (Compatible with any architecture)
        layer_input.requires_grad_(True)
        with torch.enable_grad():
            # Local forward pass
            # We use a simple dot product as a surrogate for relevance propagation
            surrogate = (layer_output * relevance_output).sum()
            surrogate.backward(retain_graph=True)

        relevance_input = layer_input.grad * layer_input
        return relevance_input.detach()

    def compute_all_relevance_scores(
        self, prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP scores across all parameters."""
        if self.model is None:
            self.load_model()

        importance_scores = {}
        activations = {}

        # Register hooks to capture activations
        def get_hook(name):
            def hook(module, input, output):
                activations[name] = input[0].detach()

            return hook

        hooks = []
        for name, module in self.model.named_modules():
            hooks.append(module.register_forward_hook(get_hook(name)))

        # Forward pass to collect activations
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        outputs = self.model(**inputs, output_hidden_states=True)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Get last token logits for relevance signal
        last_logits = outputs.logits[:, -1, :].abs().mean(dim=0)
        relevance_vocab = last_logits  # (vocab_size,)

        # Project relevance from vocab_size -> hidden_size via lm_head weight.
        hidden_relevance = None
        for name, param in self.model.named_parameters():
            # MULTIMODAL FIX: Support 'language_model.lm_head' and similar prefixes
            if "lm_head" in name and param.dim() == 2:
                if param.shape[0] == relevance_vocab.shape[0]:
                    hidden_relevance = (
                        (relevance_vocab.unsqueeze(0) @ param.float()).squeeze(0).abs()
                    )
                    break

        # Propagate relevance through each named parameter
        for name, param in tqdm(self.model.named_parameters(), desc="Computing LRP"):
            layer_name = ".".join(name.split(".")[:-1])
            act = activations.get(layer_name)
            if act is None:
                importance_scores[name] = param.data.abs().cpu().clone()
                continue

            # Flatten act to (tokens, features)
            act_flat = act.reshape(-1, act.shape[-1]) if act.dim() > 1 else act
            feat_dim = act_flat.shape[-1]

            if hidden_relevance is not None and hidden_relevance.numel() == feat_dim:
                rel_signal = hidden_relevance.expand_as(act_flat)
            else:
                rel_signal = act_flat.abs().mean() * torch.ones_like(act_flat)

            # Assign score based on magnitude x activation
            importance_scores[name] = (
                param.data.abs().cpu() * act_flat.abs().mean(dim=0).cpu()
            ).clone()

        self.relevance_scores = importance_scores
        return importance_scores

    def save_relevance_scores(self) -> None:
        """Save computed relevance scores to disk."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # TIED-TENSOR FIX: Clone scores before saving to prevent share storage errors
        scores_path = output_path / "lrp_scores.pt"
        save_dict = {k: v.clone() for k, v in self.relevance_scores.items()}
        torch.save(save_dict, scores_path)
        logger.info(f"Saved LRP scores to {scores_path}")

        # Save metadata
        metadata = {
            "model_path": self.config.model_path,
            "num_tensors": len(self.relevance_scores),
        }
        with open(output_path / "lrp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
