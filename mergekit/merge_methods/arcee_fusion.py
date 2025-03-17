# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.generalized_task_arithmetic import get_task_vectors
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes
from mergekit.sparsify import RescaleNorm, rescaled_masked_tensor


class DynamicThresholdFusion:
    def approximate_quantiles(self, tensor, q):
        # Flatten the tensor
        flat_tensor = tensor.view(-1)

        # If tensor is too large, sample it
        if flat_tensor.numel() > 1e6:
            flat_tensor = flat_tensor[torch.randperm(flat_tensor.numel())[:1000000]]

        # Sort the (possibly sampled) tensor
        sorted_tensor, _ = torch.sort(flat_tensor)

        # Compute quantile indices
        quantile_indices = (q * (sorted_tensor.numel() - 1)).long()

        # Return quantiles
        return sorted_tensor[quantile_indices]

    def calculate_dynamic_threshold(self, importance_scores):
        # Approximate median and quantiles
        median = self.approximate_quantiles(importance_scores, torch.tensor([0.5]))[0]
        q1, q3 = self.approximate_quantiles(
            importance_scores, torch.tensor([0.25, 0.75])
        )

        # Calculate IQR
        iqr = q3 - q1

        # Set threshold as median + 1.5 * IQR
        dynamic_threshold = median + 1.5 * iqr

        return dynamic_threshold

    def compute_fusion_mask(self, importance_scores):
        threshold = self.calculate_dynamic_threshold(importance_scores)
        fusion_mask = (importance_scores >= threshold).float()
        return fusion_mask, threshold


class ArceeFusionMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    normalize: bool
    ablations_kl_only: bool
    ablations_diff_only: bool
    ablations_randomise: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        rectify_embed_sizes(self.weight_info, list(tensors.values()))

        # Get base model and task vectors
        base = tensors[self.base_model]
        task_vectors, base = get_task_vectors(
            self.weight_info, self.base_model, tensors, self.tensor_parameters
        )

        fusion_task_vectors = []
        weights = []
        for task_vector in task_vectors:
            delta = task_vector["delta"]
            weights.append(task_vector["weight"])

            importance_scores = self._compute_importance(
                delta,
                base,
                self.ablations_kl_only,
                self.ablations_diff_only,
                self.ablations_randomise,
            )

            dynamic_threshold_fusion = DynamicThresholdFusion()
            fusion_mask, _threshold = dynamic_threshold_fusion.compute_fusion_mask(
                importance_scores
            )

            rescaled_masked_delta = rescaled_masked_tensor(
                delta,
                fusion_mask,
                RescaleNorm.l1 if self.rescale else None,
            )

            fusion_task_vectors.append(rescaled_masked_delta)

        if not fusion_task_vectors:
            return base

        # Stack and weight the task vectors
        deltas = torch.stack(fusion_task_vectors, dim=0)
        weights = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # Sum the weighted deltas and normalize
        mixed_delta = weighted_deltas.sum(dim=0)
        if self.normalize:
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1
            mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)

    def _compute_importance(
        self,
        params: torch.Tensor,
        base_params: torch.Tensor,
        kl_only: bool,
        diff_only: bool,
        randomise: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        diff = (params - base_params).abs()
        if randomise:
            return torch.rand_like(diff)

        if diff_only:
            return diff

        p = F.softmax(params, dim=-1) + eps
        q = F.softmax(base_params, dim=-1) + eps
        kl_div = torch.sum(p * torch.log(p / q), dim=-1)
        kl_div = kl_div.unsqueeze(-1)

        if kl_only:
            return kl_div

        return diff * kl_div

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class ArceeFusionMerge(MergeMethod):
    def name(self) -> str:
        return "arcee_fusion"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Arcee Fusion"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arcee.ai"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
            ConfigParameterDef(name="rescale", required=False, default_value=True),
            ConfigParameterDef(
                name="ablations_kl_only", required=False, default_value=False
            ),
            ConfigParameterDef(
                name="ablations_diff_only", required=False, default_value=False
            ),
            ConfigParameterDef(
                name="ablations_randomise", required=False, default_value=False
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task[torch.Tensor]:
        return ArceeFusionMergeTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            ablations_kl_only=parameters["ablations_kl_only"],
            ablations_diff_only=parameters["ablations_diff_only"],
            ablations_randomise=parameters["ablations_randomise"],
        )
