# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

"""
Model Weight Difference Analyzer

This script analyzes the weight differences between two models with the same base
architecture. It loads each model, compares corresponding layers, and reports the
percentage of differing weights and KL divergence between weight distributions.

The tool supports multiple loading strategies:
- Direct state dict comparison (most memory efficient)
- Lazy loading with Hugging Face Transformers
- Full model loading (least memory efficient)
"""

import gc
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mergekit.io.lazy_tensor_loader import LazyTensorLoader

# Constants
DEFAULT_NUM_BINS = 100
DEFAULT_EPSILON = 1e-8
SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}

# Layer type patterns for categorization
LAYER_PATTERNS = {
    "embedding": ["embed"],
    "attention": ["attn", "attention"],
    "mlp": ["mlp", "feed_forward", "ffn"],
    "norm": ["norm", "ln"],
    "output": ["lm_head", "output"],
}


class ModelComparisonError(Exception):
    """Custom exception for model comparison errors."""

    pass


def is_local_path(path: str) -> bool:
    """
    Check if a path is local or a Hugging Face model name.

    Args:
        path: Path or model name to check

    Returns:
        True if path is local, False if it's a Hugging Face model name
    """
    if os.path.exists(path):
        return True

    # Check if it looks like a Hugging Face model name (contains '/')
    if "/" in path and not path.startswith("./") and not path.startswith("/"):
        return False

    return True


def resolve_hf_model_path(model_name: str) -> str:
    """
    Resolve HF model name to cache path if it's a model name.

    Args:
        model_name: Model name or path

    Returns:
        Resolved path (cache path for HF models, original path otherwise)
    """
    if "/" in model_name and not os.path.exists(model_name):
        # Convert model name to cache path format
        cache_name = model_name.replace("/", "--")
        cache_path = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{cache_name}"
        )
        if os.path.exists(cache_path):
            return cache_path
    return model_name


def is_local_model_folder(path: str) -> bool:
    """
    Check if a path contains model files (single or sharded).

    Args:
        path: Path to check

    Returns:
        True if path contains model files
    """
    if not os.path.isdir(path):
        return False

    # Check for single model file
    if os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(
        os.path.join(path, "pytorch_model.bin")
    ):
        return True

    # Check for sharded model files
    if os.path.exists(
        os.path.join(path, "model.safetensors.index.json")
    ) or os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
        return True

    # Check for individual shard files
    for file in os.listdir(path):
        if (file.startswith("model-") and file.endswith(".safetensors")) or (
            file.startswith("pytorch_model-") and file.endswith(".bin")
        ):
            return True

    # Check for HF cache structure
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.exists(snapshots_dir):
        for snapshot in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_path):
                if (
                    os.path.exists(os.path.join(snapshot_path, "model.safetensors"))
                    or os.path.exists(os.path.join(snapshot_path, "pytorch_model.bin"))
                    or os.path.exists(
                        os.path.join(snapshot_path, "model.safetensors.index.json")
                    )
                ):
                    return True

    return False


def get_model_path(path: str) -> str:
    """
    Get the actual model path (handles HF cache structure).

    Args:
        path: Base path

    Returns:
        Path to the actual model files
    """
    # Check for direct model files
    if (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    ):
        return path

    # HF cache structure
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.exists(snapshots_dir):
        for snapshot in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_path):
                if (
                    os.path.exists(os.path.join(snapshot_path, "model.safetensors"))
                    or os.path.exists(os.path.join(snapshot_path, "pytorch_model.bin"))
                    or os.path.exists(
                        os.path.join(snapshot_path, "model.safetensors.index.json")
                    )
                ):
                    return snapshot_path
    return path


def validate_device(device: str) -> str:
    """
    Validate and normalize device string.

    Args:
        device: Device string to validate

    Returns:
        Normalized device string

    Raises:
        ValueError: If device is not supported
    """
    device = device.lower()
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{device}'. "
            f"Supported devices: {', '.join(SUPPORTED_DEVICES)}"
        )
    return device


@contextmanager
def model_context(device: str):
    """
    Context manager for loading and cleaning up models.

    Args:
        device: Device to load models on
    """
    try:
        yield
    finally:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


class ModelLoader:
    """Handles model loading with different strategies."""

    @staticmethod
    def load_lazy(
        model_name: str, device: str = "cpu"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load a model lazily (only config and tokenizer initially).

        Args:
            model_name: Name of the model to load
            device: Device to load the model on

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelComparisonError: If model loading fails
        """
        print(f"Loading model config and tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if device == "cuda" else device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            return model, tokenizer
        except Exception as e:
            raise ModelComparisonError(f"Error loading model {model_name}: {e}")

    @staticmethod
    def load_from_path_lazy(
        model_path: str, device: str = "cpu"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load a model from a local folder path lazily.

        Args:
            model_path: Path to the local model folder
            device: Device to load the model on

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelComparisonError: If model loading fails
        """
        print(f"Loading model config and tokenizer from local path: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if device == "cuda" else device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            return model, tokenizer
        except Exception as e:
            raise ModelComparisonError(f"Error loading model from {model_path}: {e}")

    @staticmethod
    def load_full(
        model_name: str, device: str = "cpu"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load a model fully (for backward compatibility).

        Args:
            model_name: Name of the model to load
            device: Device to load the model on

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelComparisonError: If model loading fails
        """
        print(f"Loading model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            return model, tokenizer
        except Exception as e:
            raise ModelComparisonError(f"Error loading model {model_name}: {e}")

    @staticmethod
    def load_from_path_full(
        model_path: str, device: str = "cpu"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load a model from a local folder path fully.

        Args:
            model_path: Path to the local model folder
            device: Device to load the model on

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelComparisonError: If model loading fails
        """
        print(f"Loading model from local path: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            return model, tokenizer
        except Exception as e:
            raise ModelComparisonError(f"Error loading model from {model_path}: {e}")


class ModelAnalyzer:
    """Handles model analysis and comparison."""

    @staticmethod
    def get_layer_names(model: nn.Module) -> List[str]:
        """
        Get all layer names from a model without loading weights.

        Args:
            model: The model to extract layer names from

        Returns:
            List of layer names
        """
        layer_names = []

        def extract_names(module: nn.Module, prefix: str = "") -> None:
            """Recursively extract layer names from all modules."""
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this is a parameter-containing module
                if list(child.parameters()):
                    for param_name, _ in child.named_parameters():
                        param_full_name = f"{full_name}.{param_name}"
                        layer_names.append(param_full_name)

                # Recursively process child modules
                extract_names(child, full_name)

        extract_names(model)
        return layer_names

    @staticmethod
    def get_single_layer_weight(model: nn.Module, layer_name: str) -> torch.Tensor:
        """
        Get a single layer's weight from a model.

        Args:
            model: The model to extract weight from
            layer_name: Name of the layer to extract

        Returns:
            The layer's weight tensor

        Raises:
            ValueError: If layer name is not found
        """
        # Navigate to the layer
        module_path = layer_name.rsplit(".", 1)[0]
        param_name = layer_name.rsplit(".", 1)[1]

        module = model
        for part in module_path.split("."):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                raise ValueError(f"Layer {layer_name} not found in model")

        if hasattr(module, param_name):
            return getattr(module, param_name).data.clone()
        else:
            raise ValueError(
                f"Parameter {param_name} not found in module {module_path}"
            )

    @staticmethod
    def get_layer_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extract all layer weights from a model.

        Args:
            model: The model to extract weights from

        Returns:
            Dictionary mapping layer names to their weights
        """
        weights = {}

        def extract_weights(module: nn.Module, prefix: str = "") -> None:
            """Recursively extract weights from all modules."""
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this is a parameter-containing module
                if list(child.parameters()):
                    for param_name, param in child.named_parameters():
                        param_full_name = f"{full_name}.{param_name}"
                        weights[param_full_name] = param.data.clone()

                # Recursively process child modules
                extract_weights(child, full_name)

        extract_weights(model)
        return weights


class WeightComparator:
    """Handles weight comparison between models."""

    @staticmethod
    def compare_weights(
        weights1: Dict[str, torch.Tensor], weights2: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compare weights between two models and calculate difference percentages.

        Args:
            weights1: Weights from first model
            weights2: Weights from second model

        Returns:
            Dictionary mapping layer names to difference percentages
        """
        differences = {}
        all_layers = set(weights1.keys()) | set(weights2.keys())

        for layer_name in all_layers:
            if layer_name in weights1 and layer_name in weights2:
                w1 = weights1[layer_name]
                w2 = weights2[layer_name]

                # Ensure shapes match
                if w1.shape != w2.shape:
                    print(
                        f"Warning: Shape mismatch for {layer_name}: {w1.shape} vs {w2.shape}"
                    )
                    differences[layer_name] = 100.0
                    continue

                # Calculate percentage of different weights efficiently
                total_elements = w1.numel()
                if total_elements == 0:
                    differences[layer_name] = 0.0
                    continue

                different_elements = torch.ne(w1, w2).sum().item()
                diff_percentage = (different_elements / total_elements) * 100
                differences[layer_name] = diff_percentage

            elif layer_name in weights1:
                print(f"Warning: Layer {layer_name} only exists in first model")
                differences[layer_name] = 100.0
            elif layer_name in weights2:
                print(f"Warning: Layer {layer_name} only exists in second model")
                differences[layer_name] = 100.0

        return differences

    @staticmethod
    def compare_weights_lazy(
        model1: nn.Module,
        model2: nn.Module,
        layer_names: List[str],
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Compare weights between two models layer by layer (memory efficient).

        Args:
            model1: First model
            model2: Second model
            layer_names: List of layer names to compare
            device: Device to load weights on

        Returns:
            Dictionary mapping layer names to difference percentages
        """
        differences = {}
        print(f"Comparing {len(layer_names)} layers...")

        for layer_name in layer_names:
            try:
                # Load weights for this layer from both models
                w1 = ModelAnalyzer.get_single_layer_weight(model1, layer_name)
                w2 = ModelAnalyzer.get_single_layer_weight(model2, layer_name)

                # Move to device if needed
                if device != "cpu":
                    w1 = w1.to(device)
                    w2 = w2.to(device)

                # Ensure shapes match
                if w1.shape != w2.shape:
                    print(
                        f"Warning: Shape mismatch for {layer_name}: {w1.shape} vs {w2.shape}"
                    )
                    differences[layer_name] = 100.0
                    continue

                # Calculate percentage of different weights efficiently
                total_elements = w1.numel()
                if total_elements == 0:
                    differences[layer_name] = 0.0
                    continue

                different_elements = torch.ne(w1, w2).sum().item()
                diff_percentage = (different_elements / total_elements) * 100
                differences[layer_name] = diff_percentage

                # Clean up memory
                del w1, w2
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Warning: Error processing layer {layer_name}: {e}")
                differences[layer_name] = 100.0

        return differences


class KLDivergenceCalculator:
    """Handles KL divergence calculation between weight distributions."""

    @staticmethod
    def compute_kl_divergence(
        weights1: Dict[str, torch.Tensor],
        weights2: Dict[str, torch.Tensor],
        num_bins: int = DEFAULT_NUM_BINS,
        epsilon: float = DEFAULT_EPSILON,
    ) -> Dict[str, float]:
        """
        Compute KL divergence between weight distributions for each layer.

        Args:
            weights1: Weights from first model
            weights2: Weights from second model
            num_bins: Number of bins for histogram computation
            epsilon: Small value to avoid division by zero

        Returns:
            Dictionary mapping layer names to KL divergence values
        """
        kl_divergences = {}
        all_layers = set(weights1.keys()) | set(weights2.keys())

        for layer_name in all_layers:
            if layer_name in weights1 and layer_name in weights2:
                w1 = weights1[layer_name]
                w2 = weights2[layer_name]

                # Ensure shapes match
                if w1.shape != w2.shape:
                    print(
                        f"Warning: Shape mismatch for {layer_name}: {w1.shape} vs {w2.shape}"
                    )
                    kl_divergences[layer_name] = float("inf")
                    continue

                kl_divergences[layer_name] = (
                    KLDivergenceCalculator._compute_single_kl_divergence(
                        w1, w2, num_bins, epsilon
                    )
                )

            elif layer_name in weights1:
                print(f"Warning: Layer {layer_name} only exists in first model")
                kl_divergences[layer_name] = float("inf")
            elif layer_name in weights2:
                print(f"Warning: Layer {layer_name} only exists in second model")
                kl_divergences[layer_name] = float("inf")

        return kl_divergences

    @staticmethod
    def compute_kl_divergence_lazy(
        model1: nn.Module,
        model2: nn.Module,
        layer_names: List[str],
        device: str = "cpu",
        num_bins: int = DEFAULT_NUM_BINS,
        epsilon: float = DEFAULT_EPSILON,
    ) -> Dict[str, float]:
        """
        Compute KL divergence between weight distributions layer by layer (memory efficient).

        Args:
            model1: First model
            model2: Second model
            layer_names: List of layer names to compare
            device: Device to load weights on
            num_bins: Number of bins for histogram computation
            epsilon: Small value to avoid division by zero

        Returns:
            Dictionary mapping layer names to KL divergence values
        """
        kl_divergences = {}
        print(f"Computing KL divergence for {len(layer_names)} layers...")

        for layer_name in layer_names:
            try:
                # Load weights for this layer from both models
                w1 = ModelAnalyzer.get_single_layer_weight(model1, layer_name)
                w2 = ModelAnalyzer.get_single_layer_weight(model2, layer_name)

                # Move to device if needed
                if device != "cpu":
                    w1 = w1.to(device)
                    w2 = w2.to(device)

                # Ensure shapes match
                if w1.shape != w2.shape:
                    print(
                        f"Warning: Shape mismatch for {layer_name}: {w1.shape} vs {w2.shape}"
                    )
                    kl_divergences[layer_name] = float("inf")
                    continue

                kl_divergences[layer_name] = (
                    KLDivergenceCalculator._compute_single_kl_divergence(
                        w1, w2, num_bins, epsilon
                    )
                )

                # Clean up memory
                del w1, w2
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Warning: Error processing layer {layer_name}: {e}")
                kl_divergences[layer_name] = float("inf")

        return kl_divergences

    @staticmethod
    def _compute_single_kl_divergence(
        w1: torch.Tensor, w2: torch.Tensor, num_bins: int, epsilon: float
    ) -> float:
        """
        Compute KL divergence between two tensors.

        Args:
            w1: First tensor
            w2: Second tensor
            num_bins: Number of bins for histogram
            epsilon: Small value to avoid division by zero

        Returns:
            KL divergence value
        """
        # Flatten tensors for histogram computation
        w1_flat = w1.flatten().cpu().numpy()
        w2_flat = w2.flatten().cpu().numpy()

        # Find global min and max for consistent binning
        global_min = min(w1_flat.min(), w2_flat.min())
        global_max = max(w1_flat.max(), w2_flat.max())

        # Handle edge case where all values are the same
        if global_min == global_max:
            return 0.0

        # Compute histograms
        hist1, _ = np.histogram(w1_flat, bins=num_bins, range=(global_min, global_max))
        hist2, _ = np.histogram(w2_flat, bins=num_bins, range=(global_min, global_max))

        # Convert to probabilities
        p1 = hist1.astype(np.float64) / (hist1.sum() + epsilon)
        p2 = hist2.astype(np.float64) / (hist2.sum() + epsilon)

        # Add small epsilon to avoid log(0)
        p1 = p1 + epsilon
        p2 = p2 + epsilon

        # Normalize again
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        # Compute KL divergence: KL(P||Q) = sum(p * log(p/q))
        # We compute both directions and take the average
        kl_forward = np.sum(p1 * np.log(p1 / p2))
        kl_backward = np.sum(p2 * np.log(p2 / p1))

        # Use symmetric KL divergence (Jensen-Shannon divergence)
        return (kl_forward + kl_backward) / 2


class ResultAnalyzer:
    """Handles analysis and presentation of comparison results."""

    @staticmethod
    def group_layers_by_type(
        differences: Dict[str, float],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Group layers by their type (e.g., attention, mlp, embedding, etc.).

        Args:
            differences: Dictionary of layer differences

        Returns:
            Dictionary grouping layers by type
        """
        grouped = {
            "embedding": [],
            "attention": [],
            "mlp": [],
            "norm": [],
            "output": [],
            "other": [],
        }

        for layer_name, diff_percentage in differences.items():
            layer_name_lower = layer_name.lower()
            categorized = False

            # Check each layer type pattern
            for layer_type, patterns in LAYER_PATTERNS.items():
                if any(pattern in layer_name_lower for pattern in patterns):
                    grouped[layer_type].append((layer_name, diff_percentage))
                    categorized = True
                    break

            # If not categorized, add to 'other'
            if not categorized:
                grouped["other"].append((layer_name, diff_percentage))

        return grouped

    @staticmethod
    def print_analysis_results(
        differences: Dict[str, float],
        kl_divergences: Optional[Dict[str, float]],
        model1_name: str,
        model2_name: str,
        verbose: bool = False,
    ):
        """
        Print the analysis results in a formatted way.

        Args:
            differences: Dictionary of layer differences
            kl_divergences: Dictionary of KL divergence values (optional)
            model1_name: Name of first model
            model2_name: Name of second model
            verbose: Whether to show detailed per-layer information
        """
        print("\n" + "=" * 80)
        print("WEIGHT DIFFERENCE ANALYSIS")
        print(f"Model 1: {model1_name}")
        print(f"Model 2: {model2_name}")
        print("=" * 80)

        # Group layers by type
        grouped = ResultAnalyzer.group_layers_by_type(differences)

        # Print summary statistics
        all_differences = list(differences.values())
        print("\nSUMMARY STATISTICS:")
        print(f"Total layers analyzed: {len(differences)}")
        print(f"Average difference: {np.mean(all_differences):.2f}%")
        print(f"Median difference: {np.median(all_differences):.2f}%")
        print(f"Min difference: {np.min(all_differences):.2f}%")
        print(f"Max difference: {np.max(all_differences):.2f}%")

        # Print KL divergence statistics if available
        if kl_divergences:
            # Filter out infinite values for statistics
            finite_kl_values = [v for v in kl_divergences.values() if v != float("inf")]
            if finite_kl_values:
                print(f"\nKL DIVERGENCE STATISTICS:")
                print(f"Average KL divergence: {np.mean(finite_kl_values):.6f}")
                print(f"Median KL divergence: {np.median(finite_kl_values):.6f}")
                print(f"Min KL divergence: {np.min(finite_kl_values):.6f}")
                print(f"Max KL divergence: {np.max(finite_kl_values):.6f}")
            else:
                print(f"\nKL DIVERGENCE: All values are infinite (shape mismatches)")

        # Print detailed results by layer type only if verbose
        if verbose:
            for layer_type, layers in grouped.items():
                if layers:
                    print(f"\n{layer_type.upper()} LAYERS:")
                    print("-" * 60)

                    # Sort by difference percentage
                    layers.sort(key=lambda x: x[1], reverse=True)

                    for layer_name, diff_percentage in layers:
                        print(f"{layer_name:<50} {diff_percentage:>8.2f}%")

            # Print layers with highest differences
            print("\nTOP 10 LAYERS WITH HIGHEST DIFFERENCES:")
            print("-" * 60)
            sorted_layers = sorted(
                differences.items(), key=lambda x: x[1], reverse=True
            )
            for layer_name, diff_percentage in sorted_layers[:10]:
                print(f"{layer_name:<50} {diff_percentage:>8.2f}%")

            # Print layers with lowest differences
            print("\nTOP 10 LAYERS WITH LOWEST DIFFERENCES:")
            print("-" * 60)
            for layer_name, diff_percentage in sorted_layers[-10:]:
                print(f"{layer_name:<50} {diff_percentage:>8.2f}%")

            # Print KL divergence information if available
            if kl_divergences:
                print("\nKL DIVERGENCE BY LAYER TYPE:")
                kl_grouped = ResultAnalyzer.group_layers_by_type(kl_divergences)

                for layer_type, layers in kl_grouped.items():
                    if layers:
                        print(f"\n{layer_type.upper()} LAYERS (KL Divergence):")
                        print("-" * 60)

                        # Sort by KL divergence
                        layers.sort(key=lambda x: x[1], reverse=True)

                        for layer_name, kl_value in layers:
                            if kl_value == float("inf"):
                                print(f"{layer_name:<50} {'INF':>8}")
                            else:
                                print(f"{layer_name:<50} {kl_value:>8.6f}")

                # Print layers with highest KL divergence
                print("\nTOP 10 LAYERS WITH HIGHEST KL DIVERGENCE:")
                print("-" * 60)
                sorted_kl_layers = sorted(
                    kl_divergences.items(), key=lambda x: x[1], reverse=True
                )
                for layer_name, kl_value in sorted_kl_layers[:10]:
                    if kl_value == float("inf"):
                        print(f"{layer_name:<50} {'INF':>8}")
                    else:
                        print(f"{layer_name:<50} {kl_value:>8.6f}")

                # Print layers with lowest KL divergence
                print("\nTOP 10 LAYERS WITH LOWEST KL DIVERGENCE:")
                print("-" * 60)
                for layer_name, kl_value in sorted_kl_layers[-10:]:
                    if kl_value == float("inf"):
                        print(f"{layer_name:<50} {'INF':>8}")
                    else:
                        print(f"{layer_name:<50} {kl_value:>8.6f}")


class DirectStateDictComparator:
    """Handles direct state dict comparison using LazyTensorLoader."""

    @staticmethod
    def compare_models(
        model_path1: str, model_path2: str, device: str, num_bins: int, verbose: bool
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compare models using direct state dict access.

        Args:
            model_path1: Path to first model
            model_path2: Path to second model
            device: Device to use
            num_bins: Number of bins for KL divergence
            verbose: Whether to show verbose output

        Returns:
            Tuple of (differences, kl_divergences)
        """
        print(
            "\nUsing direct state dict comparison (true lazy, no model instantiation)..."
        )

        loader1 = LazyTensorLoader.from_disk(model_path1)
        loader2 = LazyTensorLoader.from_disk(model_path2)
        keys1 = set(loader1.index.tensor_paths.keys())
        keys2 = set(loader2.index.tensor_paths.keys())
        all_keys = sorted(keys1 | keys2)
        print(f"Found {len(all_keys)} tensors to compare.")

        differences = {}
        kl_divergences = {}

        for key in tqdm(all_keys, desc="Comparing tensors", unit="tensor"):
            try:
                t1 = loader1.get_tensor(key, device=device, raise_on_missing=False)
                t2 = loader2.get_tensor(key, device=device, raise_on_missing=False)

                if t1 is None or t2 is None:
                    print(f"Warning: {key} only exists in one model.")
                    differences[key] = 100.0
                    kl_divergences[key] = float("inf")
                    continue

                if t1.shape != t2.shape:
                    print(
                        f"Warning: Shape mismatch for {key}: {t1.shape} vs {t2.shape}"
                    )
                    differences[key] = 100.0
                    kl_divergences[key] = float("inf")
                    continue

                # Convert to float32 if needed for compatibility
                if t1.dtype == torch.bfloat16:
                    t1 = t1.to(torch.float32)
                if t2.dtype == torch.bfloat16:
                    t2 = t2.to(torch.float32)

                # Compute difference percentage
                total_elements = t1.numel()
                if total_elements == 0:
                    differences[key] = 0.0
                    kl_divergences[key] = 0.0
                    continue

                different_elements = torch.ne(t1, t2).sum().item()
                diff_percentage = (different_elements / total_elements) * 100
                differences[key] = diff_percentage

                # Compute KL divergence
                kl_divergences[key] = (
                    KLDivergenceCalculator._compute_single_kl_divergence(
                        t1, t2, num_bins, DEFAULT_EPSILON
                    )
                )

                # Clean up memory
                del t1, t2
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Warning: Error processing tensor {key}: {e}")
                differences[key] = 100.0
                kl_divergences[key] = float("inf")

        return differences, kl_divergences


@click.command("mergekit-diff")
@click.argument("base-model", type=str)
@click.argument("model", type=str)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to load models on (cpu, cuda, mps)",
)
@click.option("--verbose", is_flag=True, help="Show detailed per-layer information")
@click.option(
    "--num-bins",
    type=int,
    default=DEFAULT_NUM_BINS,
    help="Number of bins for KL divergence histogram computation",
)
@click.option(
    "--no-lazy",
    is_flag=True,
    help="Disable lazy loading (load all weights at once, uses more memory)",
)
def main(
    base_model: str,
    model: str,
    device: str,
    verbose: bool,
    num_bins: int,
    no_lazy: bool,
):
    """
    Analyze weight differences between two models with the same base architecture.

    BASE_MODEL: Base model name or path
    MODEL: Second model name or path to compare with base model

    This tool loads each model, compares corresponding layers, and reports the
    percentage of differing weights. It also always computes the KL divergence
    between weight distributions for each layer, providing a more sophisticated
    measure of distributional differences. By default, it uses lazy loading
    (processes one layer at a time) for memory efficiency, allowing analysis of
    arbitrarily large models. Use --no-lazy to load all weights at once (uses
    more memory but may be faster for small models). Useful for understanding
    how models differ after fine-tuning, merging, or other modifications.
    """
    try:
        # Validate device
        device = validate_device(device)

        print("Analyzing weight differences between:")
        print(f"Base Model: {base_model}")
        print(f"Model: {model}")

        # Resolve HF model names to cache paths
        resolved_base_model = resolve_hf_model_path(base_model)
        resolved_model = resolve_hf_model_path(model)

        if resolved_base_model != base_model:
            print(f"Resolved base model to cache path: {resolved_base_model}")
        if resolved_model != model:
            print(f"Resolved model to cache path: {resolved_model}")

        # True PyTorch/safetensors path: both are local folders with weights
        if is_local_model_folder(resolved_base_model) and is_local_model_folder(
            resolved_model
        ):
            model_path1 = get_model_path(resolved_base_model)
            model_path2 = get_model_path(resolved_model)

            differences, kl_divergences = DirectStateDictComparator.compare_models(
                model_path1, model_path2, device, num_bins, verbose
            )

        # Otherwise, fall back to the existing approach
        elif not no_lazy:
            # Lazy loading approach - load models but not all weights (default)
            print("\nUsing lazy loading for memory efficiency...")

            # Load both models lazily
            print("Loading base model config and tokenizer...")
            with model_context(device):
                if is_local_path(base_model):
                    base_model_obj, base_tokenizer = ModelLoader.load_from_path_lazy(
                        base_model, device
                    )
                else:
                    base_model_obj, base_tokenizer = ModelLoader.load_lazy(
                        base_model, device
                    )

            print("Loading second model config and tokenizer...")
            with model_context(device):
                if is_local_path(model):
                    model2, tokenizer2 = ModelLoader.load_from_path_lazy(model, device)
                else:
                    model2, tokenizer2 = ModelLoader.load_lazy(model, device)

            # Get layer names from one model (they should be the same)
            print("Getting layer names...")
            layer_names = ModelAnalyzer.get_layer_names(base_model_obj)
            print(f"Found {len(layer_names)} layers to compare")

            # Compare weights layer by layer
            print("\nComparing weights layer by layer...")
            differences = WeightComparator.compare_weights_lazy(
                base_model_obj, model2, layer_names, device
            )

            # Always compute KL divergence
            print("\nComputing KL divergence layer by layer...")
            kl_divergences = KLDivergenceCalculator.compute_kl_divergence_lazy(
                base_model_obj, model2, layer_names, device, num_bins
            )

            # Clean up models
            del base_model_obj, base_tokenizer, model2, tokenizer2

        else:
            # Original approach - load all weights at once
            print("\nLoading base model...")
            with model_context(device):
                if is_local_path(base_model):
                    base_model_obj, base_tokenizer = ModelLoader.load_from_path_full(
                        base_model, device
                    )
                else:
                    base_model_obj, base_tokenizer = ModelLoader.load_full(
                        base_model, device
                    )

                # Extract weights from base model
                print("Extracting weights from base model...")
                base_weights = ModelAnalyzer.get_layer_weights(base_model_obj)

                # Clean up base model
                del base_model_obj, base_tokenizer

            # Load and process second model
            print("\nLoading second model...")
            with model_context(device):
                if is_local_path(model):
                    model2, tokenizer2 = ModelLoader.load_from_path_full(model, device)
                else:
                    model2, tokenizer2 = ModelLoader.load_full(model, device)

                # Extract weights from second model
                print("Extracting weights from second model...")
                model2_weights = ModelAnalyzer.get_layer_weights(model2)

                # Clean up second model
                del model2, tokenizer2

            # Compare weights
            print("\nComparing weights...")
            differences = WeightComparator.compare_weights(base_weights, model2_weights)

            # Always compute KL divergence
            print("\nComputing KL divergence...")
            kl_divergences = KLDivergenceCalculator.compute_kl_divergence(
                base_weights, model2_weights, num_bins
            )

        # Print results
        ResultAnalyzer.print_analysis_results(
            differences, kl_divergences, base_model, model, verbose
        )

    except ModelComparisonError as e:
        print(f"Error during analysis: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
