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
from mergekit.merge_methods.easy_define import from_group_kernel
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class DynamicThresholdFusion:
    def approximate_quantiles(self, tensor, q):
        flat_tensor = tensor.view(-1)
        if flat_tensor.numel() > 1e6:
            flat_tensor = flat_tensor[
                torch.randperm(flat_tensor.numel(), device=flat_tensor.device)[:1000000]
            ]
        sorted_tensor, _ = torch.sort(flat_tensor)
        quantile_indices = (
            q.to(sorted_tensor.device) * (sorted_tensor.numel() - 1)
        ).long()
        return sorted_tensor[quantile_indices]

    def calculate_dynamic_threshold(self, importance_scores):
        median = self.approximate_quantiles(importance_scores, torch.tensor([0.5]))[0]
        q1, q3 = self.approximate_quantiles(
            importance_scores, torch.tensor([0.25, 0.75])
        )
        return median + 1.5 * (q3 - q1)

    def compute_fusion_mask(self, importance_scores):
        threshold = self.calculate_dynamic_threshold(importance_scores)
        return (importance_scores >= threshold).float(), threshold


def _arcee_fusion_merge(group: TensorGroup) -> torch.Tensor:
    tensors = [group.base.tensor, group.non_base[0].tensor]
    rectify_embed_sizes(group.metadata, tensors)
    diff = (tensors[1] - tensors[0]).abs()
    eps = 1e-8
    p = F.softmax(tensors[1], dim=-1) + eps
    q = F.softmax(tensors[0], dim=-1) + eps
    kl_div = torch.sum(p * torch.log(p / q), dim=-1)
    importance = diff * kl_div.unsqueeze(-1)
    fusion_mask, _ = DynamicThresholdFusion().compute_fusion_mask(importance)
    return tensors[0] + (tensors[1] - tensors[0]) * fusion_mask


arcee_fusion_merge_method = from_group_kernel(
    _arcee_fusion_merge,
    name="arcee_fusion",
    pretty_name="Arcee Fusion",
    optional_tensor_policy=OptionalTensorPolicy.PASSTHROUGH_SINGLETON,
    reference_url="https://arcee.ai",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
