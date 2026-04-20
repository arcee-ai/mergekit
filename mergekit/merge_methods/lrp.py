# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes
from mergekit.sparsify import build_mask


class LRPMergeTask(Task[torch.Tensor]):
    """
    Performs LRP-based merging using Layer-wise Relevance Propagation scores
    to determine which weights to keep during model merging.
    """

    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    model_weights: ImmutableMap[ModelReference, float]
    density: float
    weight_info: WeightInfo
    lrp_scores: Optional[ImmutableMap[str, str]] = None  # model_ref_str -> lrp_path

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        """
        Execute LRP-based merge.

        Args:
            tensors: Dictionary mapping ModelReference to weight tensors.
                    May include LRP scores with keys suffixed by "_lrp".
        """
        # Get base tensor
        base_tensor = tensors.get(self.base_model) if self.base_model else None

        if base_tensor is None:
            first_tensor = list(tensors.values())[0] if tensors else None
            if first_tensor is None:
                raise ValueError("No tensors provided for merging")
            base_tensor = torch.zeros_like(first_tensor)

        # Collect non-base tensors
        weight_tensors = {}

        for ref, tensor in tensors.items():
            if ref == self.base_model:
                continue
            weight_tensors[ref] = tensor

        if not weight_tensors:
            return base_tensor

        # Rectify embedding sizes - operate on a list so all tensors are updated
        refs = list(weight_tensors.keys())
        all_tensors = [base_tensor] + [weight_tensors[r] for r in refs]
        rectify_embed_sizes(self.weight_info, all_tensors)

        # Propagate rectified tensors back to base and weight_tensors
        base_tensor = all_tensors[0]
        weight_tensors = {r: all_tensors[i + 1] for i, r in enumerate(refs)}

        # Initialize merged deltas
        merged_deltas = torch.zeros_like(base_tensor)

        # Validate weights
        if not self.model_weights:
            raise ValueError("model_weights cannot be empty")

        total_weight = sum(self.model_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of model weights cannot be zero")

        # Process each model
        _lrp_cache: Dict[str, Any] = {}
        for ref, fine_tuned_weight in weight_tensors.items():
            # Validate tensor shape
            if fine_tuned_weight.shape != base_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {ref}: expected {base_tensor.shape}, got {fine_tuned_weight.shape}"
                )

            # Compute delta (task vector)
            delta = fine_tuned_weight - base_tensor

            # Get LRP importance scores if available from lrp_scores parameter
            importance = None
            ref_str = str(ref)
            if self.lrp_scores is not None and ref_str in self.lrp_scores:
                lrp_path = self.lrp_scores[ref_str]
                if lrp_path not in _lrp_cache:
                    _lrp_cache[lrp_path] = torch.load(lrp_path, map_location="cpu")
                importance = _lrp_cache[lrp_path].get(self.weight_info.name)
                if importance is not None:
                    importance = importance.to(delta.device)

            # Fallback to magnitude-based importance
            if importance is None:
                importance = delta.abs()

            # Validate importance shape
            if importance.shape != delta.shape:
                importance = delta.abs()

            # Sparsify based on importance
            mask = build_mask(importance, self.density)
            sparse_delta = delta * mask

            # Weighted averaging
            weight = self.model_weights[ref] if ref in self.model_weights else 1.0
            normalized_weight = weight / total_weight
            merged_deltas += normalized_weight * sparse_delta

        # Final merged tensor
        return base_tensor + merged_deltas

    def uses_accelerator(self) -> bool:
        return True

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    def priority(self) -> int:
        return 0


class LRPMerge(MergeMethod):
    """
    LRP-based merge method using Layer-wise Relevance Propagation scores.

    Merges fine-tuned models by:
    1. Computing task vectors (deltas from base)
    2. Using LRP importance scores to determine which weights are most relevant
    3. Sparsifying based on importance (LRP scores or magnitude fallback)
    4. Weighted averaging of sparse deltas
    """

    @override
    def name(self) -> str:
        return "lrp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "LRP Merge"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://github.com/arcee-ai/mergekit"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", required=False, default_value=0.7),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=False, default_value=1.0)]

    @override
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        lrp_scores: Optional[Dict[str, str]] = None,
        **_kwargs,
    ) -> Task:
        """Create the LRP merge task with proper validation."""
        # Collect model weights from non-base models
        model_weights = {}
        for model_ref, params in tensor_parameters.items():
            if model_ref != base_model:
                try:
                    weight = params["weight"]
                except (KeyError, TypeError):
                    weight = 1.0
                model_weights[model_ref] = weight

        if not model_weights:
            raise ValueError(
                "At least one fine-tuned model (other than base) is required for LRP merge"
            )

        # Get density parameter
        try:
            density = parameters["density"]
        except (KeyError, TypeError):
            density = 0.7

        # Validate density
        if not 0 <= density <= 1:
            raise ValueError(f"density must be between 0 and 1, got {density}")

        return LRPMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            model_weights=ImmutableMap(model_weights),
            density=density,
            weight_info=output_weight,
            lrp_scores=ImmutableMap(lrp_scores) if lrp_scores else None,
        )
