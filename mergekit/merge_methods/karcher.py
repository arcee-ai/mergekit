# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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


class KarcherTask(Task[torch.Tensor]):
    """
    Task for merging model weights using the Riemannian (Karcher) mean algorithm.

    The Karcher mean provides a geometrically meaningful way to average points on a manifold,
    which is particularly useful for merging model weights that can be interpreted as points
    on a hypersphere.
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    max_iter: int
    tol: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # Extract tensors and prepare for merging
        model_tensors = list(tensors.values())

        # Ensure all tensors have compatible shapes
        for i in range(1, len(model_tensors)):
            rectify_embed_sizes(self.weight_info, [model_tensors[0], model_tensors[i]])

        # Calculate weights (equal by default)
        alphas = [1.0 / len(model_tensors)] * len(model_tensors)

        # Apply Karcher mean algorithm
        return karcher_merge_tensors(
            model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
        )

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class KarcherMerge(MergeMethod):
    """
    Implementation of the Karcher mean merge method.

    This method merges model weights using the Riemannian (Karcher) mean concept,
    which provides a geometrically meaningful way to average points on a manifold.
    """

    def name(self) -> str:
        return "karcher"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Karcher Mean"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://en.wikipedia.org/wiki/Karcher_mean"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="max_iter", required=False, default_value=10),
            ConfigParameterDef(name="tol", required=False, default_value=1e-5),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        # Use default values from parameters() if not provided
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 10
        tol = parameters["tol"] if "tol" in parameters else 1e-5

        return KarcherTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            max_iter=max_iter,
            tol=tol,
        )


def karcher_merge_tensors(tensors, alphas, max_iter=10, tol=1e-5):
    """
    Implements weight fusion based on the Riemannian (Karcher) mean concept.

    Args:
        tensors: List of tensors to merge
        alphas: List of weights for each tensor
        max_iter: Maximum number of iterations for the Karcher mean algorithm
        tol: Convergence tolerance

    Returns:
        Merged tensor using Karcher mean algorithm
    """
    if len(tensors) == 1:
        return tensors[0]

    # Calculate norms and unit vectors
    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))

    # Select non-zero weight vectors
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: Normalized weighted arithmetic mean
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    # Iterative Karcher mean computation
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Flatten tensor for dot product calculation
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            theta_val = theta.item()
            if theta_val < tol:
                continue
            else:
                # Ensure tensor operations
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)

        # Convert norm_T to tensor
        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break

        # Use tensor for trigonometric calculations
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)

        # Ensure u is a unit vector
        u_norm = torch.linalg.norm(u.float())
        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)

    # Global scale: Weighted sum of original tensor norms (including zero vectors)
    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n

    return s * u
