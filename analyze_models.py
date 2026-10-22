#!/usr/bin/env python3
"""
Model Weight Difference Analyzer

This script analyzes the weight differences between two models with the same base
architecture. It loads each model, compares corresponding layers, and reports the
percentage of differing weights.
"""

import argparse
import gc
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_local_path(path: str) -> bool:
    """
    Check if a path is local or a Hugging Face model name.

    Args:
        path: Path or model name to check

    Returns:
        True if path is local, False if it's a Hugging Face model name
    """
    # Check if it's a local file or directory
    if os.path.exists(path):
        return True

    # Check if it looks like a Hugging Face model name (contains '/')
    if "/" in path and not path.startswith("./") and not path.startswith("/"):
        return False

    # Default to local path if it doesn't exist
    return True


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


def load_model(model_name: str, device: str = "cpu") -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load a model and its tokenizer from Hugging Face.

    Args:
        model_name: Name of the model to load
        device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        Exception: If model loading fails
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
        print(f"Error loading model {model_name}: {e}")
        raise


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

    # Get all unique layer names
    all_layers = set(weights1.keys()) | set(weights2.keys())

    for layer_name in all_layers:
        if layer_name in weights1 and layer_name in weights2:
            w1 = weights1[layer_name]
            w2 = weights2[layer_name]

            # Ensure shapes match
            if w1.shape != w2.shape:
                print(
                    f"Warning: Shape mismatch for {layer_name}: "
                    f"{w1.shape} vs {w2.shape}"
                )
                # 100% different due to shape mismatch
                differences[layer_name] = 100.0
                continue

            # Calculate percentage of different weights efficiently
            total_elements = w1.numel()
            if total_elements == 0:
                differences[layer_name] = 0.0
                continue

            # Use torch.ne() for better performance
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

    # Define layer type patterns for better maintainability
    layer_patterns = {
        "embedding": ["embed"],
        "attention": ["attn", "attention"],
        "mlp": ["mlp", "feed_forward", "ffn"],
        "norm": ["norm", "ln"],
        "output": ["lm_head", "output"],
    }

    for layer_name, diff_percentage in differences.items():
        layer_name_lower = layer_name.lower()
        categorized = False

        # Check each layer type pattern
        for layer_type, patterns in layer_patterns.items():
            if any(pattern in layer_name_lower for pattern in patterns):
                grouped[layer_type].append((layer_name, diff_percentage))
                categorized = True
                break

        # If not categorized, add to 'other'
        if not categorized:
            grouped["other"].append((layer_name, diff_percentage))

    return grouped


def print_analysis_results(
    differences: Dict[str, float],
    model1_name: str,
    model2_name: str,
    verbose: bool = False,
):
    """
    Print the analysis results in a formatted way.

    Args:
        differences: Dictionary of layer differences
        model1_name: Name of first model
        model2_name: Name of second model
    """
    print("\n" + "=" * 80)
    print("WEIGHT DIFFERENCE ANALYSIS")
    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")
    print("=" * 80)

    # Group layers by type
    grouped = group_layers_by_type(differences)

    # Print summary statistics
    all_differences = list(differences.values())
    print("\nSUMMARY STATISTICS:")
    print(f"Total layers analyzed: {len(differences)}")
    print(f"Average difference: {np.mean(all_differences):.2f}%")
    print(f"Median difference: {np.median(all_differences):.2f}%")
    print(f"Min difference: {np.min(all_differences):.2f}%")
    print(f"Max difference: {np.max(all_differences):.2f}%")

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
        sorted_layers = sorted(differences.items(), key=lambda x: x[1], reverse=True)
        for layer_name, diff_percentage in sorted_layers[:10]:
            print(f"{layer_name:<50} {diff_percentage:>8.2f}%")

        # Print layers with lowest differences
        print("\nTOP 10 LAYERS WITH LOWEST DIFFERENCES:")
        print("-" * 60)
        for layer_name, diff_percentage in sorted_layers[-10:]:
            print(f"{layer_name:<50} {diff_percentage:>8.2f}%")


def load_model_from_path(
    model_path: str, device: str = "cpu"
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load a model from a local folder path.

    Args:
        model_path: Path to the local model folder
        device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        Exception: If model loading fails
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
        print(f"Error loading model from {model_path}: {e}")
        raise


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
    supported_devices = {"cpu", "cuda", "mps"}

    if device not in supported_devices:
        raise ValueError(
            f"Unsupported device '{device}'. "
            f"Supported devices: {', '.join(supported_devices)}"
        )

    return device


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weight differences between models"
    )
    parser.add_argument(
        "--base-model", type=str, required=True, help="Base model name or path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Second model name or path to compare with base model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed per-layer information"
    )

    args = parser.parse_args()

    # Validate device
    try:
        args.device = validate_device(args.device)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("Analyzing weight differences between:")
    print(f"Base Model: {args.base_model}")
    print(f"Model: {args.model}")

    try:
        # Load and process base model
        print("\nLoading base model...")
        with model_context(args.device):
            if is_local_path(args.base_model):
                base_model, base_tokenizer = load_model_from_path(
                    args.base_model, args.device
                )
            else:
                base_model, base_tokenizer = load_model(args.base_model, args.device)

            # Extract weights from base model
            print("Extracting weights from base model...")
            base_weights = get_layer_weights(base_model)

            # Clean up base model
            del base_model, base_tokenizer

        # Load and process second model
        print("\nLoading second model...")
        with model_context(args.device):
            if is_local_path(args.model):
                model2, tokenizer2 = load_model_from_path(args.model, args.device)
            else:
                model2, tokenizer2 = load_model(args.model, args.device)

            # Extract weights from second model
            print("Extracting weights from second model...")
            model2_weights = get_layer_weights(model2)

            # Clean up second model
            del model2, tokenizer2

        # Compare weights
        print("\nComparing weights...")
        differences = compare_weights(base_weights, model2_weights)

        # Print results
        print_analysis_results(differences, args.base_model, args.model, args.verbose)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
