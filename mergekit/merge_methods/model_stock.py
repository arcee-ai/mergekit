# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    OptionalTensorPolicy,
    Shared,
    TensorGroup,
)
from mergekit.merge_methods.easy_define import from_group_kernel


def _model_stock_merge(
    group: TensorGroup, filter_wise: Shared[bool] = False
) -> torch.Tensor:
    all_weights = [group.base.tensor] + [entry.tensor for entry in group.non_base]
    w_0, ws = all_weights[0], all_weights[1:]
    out_shape = w_0.shape

    if filter_wise:
        if w_0.dim() == 1:
            w_0 = w_0.unsqueeze(0)
            ws = [weight.unsqueeze(0) for weight in ws]
    else:
        w_0 = w_0.reshape(-1)
        ws = [weight.reshape(-1) for weight in ws]

    offsets = [weight - w_0 for weight in ws]
    cos_thetas = []
    for idx, offset_a in enumerate(offsets):
        for offset_b in offsets[idx + 1 :]:
            norm_product = torch.norm(offset_a, dim=-1) * torch.norm(offset_b, dim=-1)
            cos_thetas.append(
                (
                    (offset_a * offset_b).sum(dim=-1) / norm_product.clamp(min=1e-6)
                ).clamp(-1, 1)
            )

    cos_theta = torch.stack(cos_thetas).mean(dim=0).unsqueeze(-1)
    count = len(ws)
    denominator = 1 + (count - 1) * cos_theta
    # At the singularity there is no finite interpolation estimate. Keep the
    # base for that filter instead of amplifying opposing updates. Mask the
    # denominator too: torch.where alone would leave NaNs in backward().
    singular = denominator.abs() < 1e-6
    safe_denominator = torch.where(singular, torch.ones_like(denominator), denominator)
    t = torch.where(
        singular,
        torch.zeros_like(cos_theta),
        (count * cos_theta) / safe_denominator,
    )
    average = sum(ws) / count
    return (t * average + (1 - t) * w_0).reshape(out_shape)


model_stock_merge_method = from_group_kernel(
    _model_stock_merge,
    name="model_stock",
    pretty_name="Model Stock",
    optional_tensor_policy=OptionalTensorPolicy.BASE_OR_SKIP,
    reference_url="https://arxiv.org/abs/2403.19522",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=3,
        min_non_base=2,
    ),
)
