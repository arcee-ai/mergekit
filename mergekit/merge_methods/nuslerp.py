# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    OptionalTensorPolicy,
    PerNonBase,
    TensorGroup,
)
from mergekit.merge_methods.easy_define import merge_method


def nuslerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    flatten: bool = False,
):
    out_shape = v0.shape

    def _normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    if flatten:
        v0 = v0.reshape(-1)
        v1 = v1.reshape(-1)
    elif dim != -1:
        v0 = v0.transpose(dim, -1)
        v1 = v1.transpose(dim, -1)

    v0_u = _normalize(v0)
    v1_u = _normalize(v1)
    cos_theta = torch.sum(v0_u * v1_u, dim=-1, keepdim=True).clamp(-1, 1)
    colinear = (1 - cos_theta.square()) <= eps**2
    # Keep the unselected spherical branch finite, including its derivatives.
    # Overwriting a singular result afterward still leaves NaNs in backward().
    theta = torch.acos(torch.where(colinear, 0, cos_theta))
    sin_theta = torch.sin(theta)

    result = (torch.sin((1 - t) * theta) * v0 + torch.sin(t * theta) * v1) / sin_theta
    result = torch.where(colinear, (1 - t) * v0 + t * v1, result)

    if dim != -1 and not flatten:
        result = result.transpose(dim, -1)
    return result.reshape(out_shape)


def _nuslerp_merge(
    group: TensorGroup,
    weight: PerNonBase[float],
    nuslerp_row_wise: bool = False,
    nuslerp_flatten: bool = True,
) -> torch.Tensor:
    entries = group.non_base
    tensors = [entry.tensor for entry in entries]
    weights = weight.values_for(entries)
    total = sum(weights)
    t = 0.5 if abs(total) < 1e-6 else weights[1] / total
    base_tensor = group.base.tensor if group.base else None
    if base_tensor is not None:
        return base_tensor + nuslerp(
            t,
            tensors[0] - base_tensor,
            tensors[1] - base_tensor,
            dim=0 if nuslerp_row_wise else -1,
            flatten=nuslerp_flatten,
        )
    return nuslerp(
        t,
        tensors[0],
        tensors[1],
        dim=0 if nuslerp_row_wise else -1,
        flatten=nuslerp_flatten,
    )


nuslerp_merge_method = merge_method(
    _nuslerp_merge,
    name="nuslerp",
    pretty_name="NuSLERP",
    optional_tensor_policy=OptionalTensorPolicy.PASSTHROUGH_SINGLETON,
    contract=InputContract(
        base=BasePolicy.OPTIONAL,
        min_non_base=2,
        max_non_base=2,
    ),
)
