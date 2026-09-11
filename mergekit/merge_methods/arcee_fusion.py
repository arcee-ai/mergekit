# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch
import torch.nn.functional as F

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    OptionalTensorPolicy,
    TensorGroup,
)
from mergekit.merge_methods.easy_define import merge_method

_QUANTILE_SAMPLE_SIZE = 1_000_000


class DynamicThresholdFusion:
    def approximate_quantiles(self, tensor, q):
        flat_tensor = tensor.reshape(-1)
        if flat_tensor.numel() > _QUANTILE_SAMPLE_SIZE:
            # Sampling with replacement bounds index storage by the sample size.
            flat_tensor = flat_tensor[
                torch.randint(
                    flat_tensor.numel(),
                    (_QUANTILE_SAMPLE_SIZE,),
                    device=flat_tensor.device,
                )
            ]
        sorted_tensor, _ = torch.sort(flat_tensor)
        quantile_indices = (
            q.to(sorted_tensor.device) * (sorted_tensor.numel() - 1)
        ).long()
        return sorted_tensor[quantile_indices]

    def calculate_dynamic_threshold(self, importance_scores):
        q1, median, q3 = self.approximate_quantiles(
            importance_scores, torch.tensor([0.25, 0.5, 0.75])
        )
        return median + 1.5 * (q3 - q1)

    def compute_fusion_mask(self, importance_scores):
        threshold = self.calculate_dynamic_threshold(importance_scores)
        return (importance_scores >= threshold).float(), threshold


def _compute_importance(params: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Keep full-sized inference temporaries out of the threshold/fusion phase."""
    diff = (params - base).abs()
    eps = 1e-8
    p = F.softmax(params, dim=-1) + eps
    q = F.softmax(base, dim=-1) + eps
    kl_div = torch.sum(p * torch.log(p / q), dim=-1, keepdim=True)
    return diff * kl_div


def _arcee_fusion_merge(group: TensorGroup) -> torch.Tensor:
    tensors = [group.base.tensor, group.non_base[0].tensor]
    importance = _compute_importance(tensors[1], tensors[0])
    fusion_mask, _ = DynamicThresholdFusion().compute_fusion_mask(importance)
    return tensors[0] + (tensors[1] - tensors[0]) * fusion_mask


arcee_fusion_merge_method = merge_method(
    _arcee_fusion_merge,
    name="arcee_fusion",
    pretty_name="Arcee Fusion",
    optional_tensor_policy=OptionalTensorPolicy.PASSTHROUGH_SINGLETON,
    reference_url="https://arcee.ai",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
    ),
)
