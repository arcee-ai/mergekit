# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional

import torch
from torch._tensor import Tensor
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


class NuSlerpTask(Task[torch.Tensor]):
    """Task for performing NuSLERP or ChipAlign merges between two model tensors.
    
    Supports both traditional NuSLERP and ChipAlign-style geodesic interpolation
    with magnitude preservation, as described in https://arxiv.org/abs/2412.19819.
    """
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    row_wise: bool
    flatten: bool
    base_model: Optional[ModelReference]
    geodesic: bool  # Whether to use ChipAlign-style geodesic interpolation
    lambda_val: Optional[float]  # Interpolation factor for geodesic mode

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        # Fast path for single-model case
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # Handle base model if provided
        if self.base_model is not None:
            if len(tensors) != 3:
                raise RuntimeError(
                    "NuSlerp base model can not be one of the two models to merge"
                )
            base_tensor = tensors.pop(self.base_model)
        else:
            base_tensor = None

        # Extract tensors and weights
        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        # Verify exactly two models are provided
        if len(tensors) != 2:
            raise RuntimeError(
                "NuSlerp merge expects exactly two models (plus optional base model)"
            )

        # Calculate interpolation factor from weights
        if abs(sum(weights)) < 1e-6:
            t = 0.5  # Default when weights sum to zero
        else:
            t = weights[1] / sum(weights)

        # Handle embedding tensors with different sizes
        if base_tensor is not None:
            tensors.append(base_tensor)
        rectify_embed_sizes(self.weight_info, tensors)

        # ChipAlign geodesic interpolation path
        if self.geodesic:
            if base_tensor is not None:
                raise ValueError("ChipAlign-style geodesic interpolation does not support a base model.")
            if self.lambda_val is None:
                raise ValueError("lambda must be specified when geodesic=True")
            
            # Extract the instruction and domain-specific tensors
            instruction_tensor = tensors[0]
            domain_tensor = tensors[1]

            # Calculate norms for magnitude preservation
            instruction_tensor_norm = torch.norm(instruction_tensor)
            domain_tensor_norm = torch.norm(domain_tensor)
            
            # Normalize to unit vectors
            instruction_tensor_unit = instruction_tensor / instruction_tensor_norm
            domain_tensor_unit = domain_tensor / domain_tensor_norm

            # Perform spherical interpolation on unit vectors
            from mergekit.merge_methods.slerp import slerp
            merged_tensor_unit = slerp(
                self.lambda_val, instruction_tensor_unit, domain_tensor_unit
            )

            # Apply magnitude scaling using weighted geometric mean (from ChipAlign paper)
            merged_tensor = (
                (instruction_tensor_norm ** (1 - self.lambda_val))
                * (domain_tensor_norm ** self.lambda_val)
                * merged_tensor_unit
            )
            return merged_tensor
        
        # Standard NuSlerp path
        if base_tensor is not None:
            base_tensor = tensors.pop()
            # For task vector mode (with base model)
            return base_tensor + nuslerp(
                t,
                tensors[0] - base_tensor,
                tensors[1] - base_tensor,
                dim=0 if self.row_wise else -1,
                flatten=self.flatten,
            )
        
        # Direct tensor mode (no base model)
        return nuslerp(
            t,
            tensors[0],
            tensors[1],
            dim=0 if self.row_wise else -1,
            flatten=self.flatten,
        )


class NuSlerpMerge(MergeMethod):
    """Merge method implementing both NuSLERP and ChipAlign geodesic interpolation.
    
    Provides a flexible, enhanced implementation of spherical linear interpolation
    with additional options for interpolation mode and parameter customization.
    """
    def name(self) -> str:
        return "nuslerp"

    @override
    def pretty_name(self):
        return "NuSLERP"

    @override
    def reference_url(self):
        return "https://arxiv.org/abs/2412.19819" if self.is_chipalign() else None
    
    def is_chipalign(self) -> bool:
        """Check if configured as ChipAlign mode based on parameters."""
        try:
            return self._parameters and self._parameters.get("geodesic", False)
        except AttributeError:
            return False

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="nuslerp_row_wise",
                required=False,
                default_value=False,
                description="SLERP row vectors instead of column vectors",
            ),
            ConfigParameterDef(
                name="nuslerp_flatten",
                required=False,
                default_value=True,
                description="Treat tensors as flattened vectors",
            ),
            ConfigParameterDef(
                name="geodesic",
                required=False,
                default_value=False,
                description="Enable ChipAlign-style geodesic interpolation with magnitude preservation",
            ),
            ConfigParameterDef(
                name="lambda",
                required=False,
                default_value=None,
                description="Interpolation factor (0.0-1.0) for geodesic mode; 0=first model, 1=second model",
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        # Store parameters for reference_url to detect ChipAlign mode
        self._parameters = parameters
        
        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            row_wise=parameters["nuslerp_row_wise"],
            flatten=parameters["nuslerp_flatten"],
            base_model=base_model,
            geodesic=parameters["geodesic"],
            lambda_val=parameters["lambda"],
        )


def nuslerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    flatten: bool = False,
):
    """Enhanced spherical linear interpolation (SLERP) with flexible tensor handling.
    
    Args:
        t: Interpolation factor between 0.0 and 1.0
        v0: First tensor
        v1: Second tensor
        dim: Dimension along which to perform row/column-wise interpolation
        eps: Small value to prevent division by zero
        flatten: Whether to flatten tensors before interpolation
    
    Returns:
        Interpolated tensor with the same shape as inputs
    """
    out_shape = v0.shape

    def _normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Normalize tensor along last dimension with numeric stability."""
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    # Handle tensor reshaping based on interpolation mode
    if flatten:
        # Treat entire tensor as a single vector
        v0 = v0.view(-1)
        v1 = v1.view(-1)
    elif dim != -1:
        # Perform interpolation along specified dimension
        v0 = v0.transpose(dim, -1)
        v1 = v1.transpose(dim, -1)

    # Normalize to unit vectors
    v0_u = _normalize(v0)
    v1_u = _normalize(v1)

    # Calculate angle between vectors
    cos_theta = torch.sum(v0_u * v1_u, dim=-1, keepdim=True)
    theta = torch.acos(cos_theta.clamp(-1, 1))
    sin_theta = torch.sin(theta)

    # Handle (nearly) colinear vectors to avoid numerical issues
    colinear = (sin_theta.abs() < eps).squeeze()

    # SLERP formula: (sin((1-t)*θ)/sin(θ))*v0 + (sin(t*θ)/sin(θ))*v1
    res = (torch.sin((1 - t) * theta) * v0 + torch.sin(t * theta) * v1) / sin_theta
    
    # Fall back to linear interpolation for numerically colinear vectors
    res[colinear] = (1 - t) * v0[colinear] + t * v1[colinear]

    # Restore original tensor shape
    if dim != -1 and not flatten:
        res = res.transpose(dim, -1)
    return res.view(out_shape)
