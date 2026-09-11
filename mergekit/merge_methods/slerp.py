# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Union

import numpy as np
import torch

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    Shared,
    TensorGroup,
    method_from_function,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """Spherical linear interpolation."""
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)
    dot = np.sum(v0 * v1)

    if np.abs(dot) > DOT_THRESHOLD:
        return maybe_torch(lerp(t, v0_copy, v1_copy), is_torch)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    return maybe_torch(s0 * v0_copy + s1 * v1_copy, is_torch)


def maybe_torch(v: np.ndarray, is_torch: bool):
    return torch.from_numpy(v) if is_torch else v


def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    return v / norm_v if norm_v > eps else v


def _slerp_merge(group: TensorGroup, t: Shared[float]) -> torch.Tensor:
    base = group.base.tensor
    other = group.non_base[0].tensor
    tensors = [base, other]
    rectify_embed_sizes(group.metadata, tensors)
    return slerp(t, tensors[0], tensors[1]).to(base.dtype).to(base.device)


slerp_merge_method = method_from_function(
    _slerp_merge,
    name="slerp",
    pretty_name="SLERP",
    reference_url="https://en.wikipedia.org/wiki/Slerp",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
