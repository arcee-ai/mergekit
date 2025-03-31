# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


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

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError("ArceeFusion merge expects exactly two models")
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        prepped_tensors = [a[1], b[1]]

        rectify_embed_sizes(self.weight_info, prepped_tensors)

        importance_scores = self._compute_importance(
            prepped_tensors[1], prepped_tensors[0]
        )
        dynamic_threshold_fusion = DynamicThresholdFusion()
        fusion_mask, _threshold = dynamic_threshold_fusion.compute_fusion_mask(
            importance_scores
        )

        delta = prepped_tensors[1] - prepped_tensors[0]
        masked_delta = delta * fusion_mask
        fused = prepped_tensors[0] + masked_delta

        return fused

    def _compute_importance(
        self, params: torch.Tensor, base_params: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        diff = (params - base_params).abs()
        p = F.softmax(params, dim=-1) + eps
        q = F.softmax(base_params, dim=-1) + eps
        kl_div = torch.sum(p * torch.log(p / q), dim=-1)
        return diff * kl_div.unsqueeze(-1)


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
        return []

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        **kwargs,
    ) -> Task[torch.Tensor]:
        return ArceeFusionMergeTask(
            gather_tensors=tensors, weight_info=output_weight, base_model=base_model
        )
