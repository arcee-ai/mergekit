# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import gc
import logging
import time
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

_TASK_COUNTER = 0
logger = logging.getLogger(__name__)


class LRPMergeTask(Task[torch.Tensor], frozen=True):
    """
    Supercharged LRP Merge Task:
    - Multimodal support (optional tensor handling)
    - Turbo optimizations (Iron-Man stabilization, in-place math)
    - Resource-efficient (GC management for Windows)
    """

    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    model_weights: ImmutableMap[ModelReference, float]
    density: float
    weight_info: WeightInfo
    lrp_scores: Optional[ImmutableMap[str, str]] = None

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        global _TASK_COUNTER
        _TASK_COUNTER += 1

        # IRON-MAN STABILIZATION: Prevent SSD/Memory thrashing on Windows
        if _TASK_COUNTER % 100 == 0:
            gc.collect()
            time.sleep(1.0)

        # Get base tensor
        base_tensor = tensors.get(self.base_model) if self.base_model else None

        if base_tensor is None:
            first_tensor = next(iter(tensors.values())) if tensors else None
            if first_tensor is None:
                raise ValueError("No tensors provided for merging")
            base_tensor = torch.zeros_like(first_tensor)

        # Collect and rectify non-base tensors
        weight_tensors = {
            ref: t for ref, t in tensors.items() if ref != self.base_model
        }
        if not weight_tensors:
            return base_tensor

        # Rectification for embedding size mismatches
        refs = list(weight_tensors.keys())
        all_tensors = [base_tensor] + [weight_tensors[r] for r in refs]
        rectify_embed_sizes(self.weight_info, all_tensors)
        base_tensor = all_tensors[0]
        weight_tensors = {r: all_tensors[i + 1] for i, r in enumerate(refs)}

        # Initialize merged deltas (using in-place additions later)
        merged_deltas = torch.zeros_like(base_tensor)

        total_weight = sum(self.model_weights.values())
        if total_weight == 0:
            total_weight = 1.0

        _lrp_cache: Dict[str, Any] = {}

        # CRITICAL LAYER PROTECTION: 100% density for norms, heads, etc.
        name = self.weight_info.name.lower()
        is_critical = any(x in name for x in ["norm", "embed", "ln_", "head", "bias"])
        current_density = 1.0 if is_critical else self.density

        for ref, fine_tuned_weight in weight_tensors.items():
            # MULTIMODAL FIX: Handle optional tensors (dt_bias, etc.) in hybrid architectures
            if fine_tuned_weight is None:
                continue

            # Validate tensor shape
            if fine_tuned_weight.shape != base_tensor.shape:
                continue

            # Compute delta (task vector) - in-place subtraction to save memory
            delta = fine_tuned_weight.sub(base_tensor)

            importance = None
            ref_str = str(ref)
            if self.lrp_scores is not None and ref_str in self.lrp_scores:
                lrp_path = self.lrp_scores[ref_str]
                if lrp_path not in _lrp_cache:
                    try:
                        if lrp_path.endswith(".safetensors"):
                            from safetensors.torch import load_file

                            _lrp_cache[lrp_path] = load_file(lrp_path)
                        else:
                            _lrp_cache[lrp_path] = torch.load(
                                lrp_path, map_location="cpu"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load LRP scores from {lrp_path}: {e}"
                        )
                        _lrp_cache[lrp_path] = {}
                importance = _lrp_cache[lrp_path].get(self.weight_info.name)
                if importance is not None:
                    importance = importance.to(delta.device)

            if importance is None:
                # MULTIMODAL FIX: Treat missing importance as passthrough for this model
                continue

            if importance.shape != delta.shape:
                importance = delta.abs()

            # Sparsify based on importance
            mask = build_mask(importance, current_density)

            # Weighted addition to merged_deltas
            weight = self.model_weights.get(ref, 1.0)
            normalized_weight = weight / total_weight

            # In-place accumulation to save memory
            merged_deltas.add_(delta.mul_(mask), alpha=normalized_weight)

            # Clean up to keep memory footprint low
            del delta, mask, importance
            if _TASK_COUNTER % 10 == 0:
                gc.collect()

        return base_tensor.add_(merged_deltas)

    def uses_accelerator(self) -> bool:
        return True

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    def priority(self) -> int:
        return 0


class LRPMerge(MergeMethod):
    """
    Explainable Layer-wise Relevance Propagation (eX-LRP) Merge Method.
    Optimized for multimodal support and high-performance execution.
    """

    @override
    def name(self) -> str:
        return "lrp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "LRP Merge (Supercharged)"

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
        model_weights = {}
        for model_ref, params in tensor_parameters.items():
            if model_ref != base_model:
                model_weights[model_ref] = params.get("weight", 1.0)

        density = parameters.get("density", 0.7)
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
