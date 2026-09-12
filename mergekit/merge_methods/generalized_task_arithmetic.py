# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Optional

import torch
from typing_extensions import Literal

from mergekit.merge_methods.base import (
    BasePolicy,
    GroupMergeMethod,
    InputContract,
    MergeMethodSpec,
    ParameterScope,
    ParameterSpec,
    TensorGroup,
)
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


@dataclass(frozen=True)
class GeneralizedTaskArithmeticMerge(GroupMergeMethod):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    @cached_property
    def spec(self) -> MergeMethodSpec:
        params = [
            ParameterSpec("int8_mask", bool, ParameterScope.SHARED, default=False),
            ParameterSpec(
                "normalize",
                bool,
                ParameterScope.SHARED,
                default=self.default_normalize,
            ),
            ParameterSpec(
                "rescale", bool, ParameterScope.SHARED, default=self.default_rescale
            ),
            ParameterSpec("lambda", float, ParameterScope.SHARED, default=1.0),
            ParameterSpec(
                "weight",
                float,
                ParameterScope.NON_BASE,
            ),
            ParameterSpec(
                "density",
                float,
                ParameterScope.NON_BASE,
                default=1.0,
            ),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            params.append(
                ParameterSpec(
                    "gamma",
                    float,
                    ParameterScope.NON_BASE,
                    default=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.della_magprune:
            params.append(
                ParameterSpec(
                    "epsilon",
                    float,
                    ParameterScope.NON_BASE,
                    default=0.15,
                )
            )
        return MergeMethodSpec(
            name=self.method_name,
            pretty_name=self.method_pretty_name,
            reference_url=self.method_reference_url,
            parameters=tuple(params),
            contract=InputContract(base=BasePolicy.REQUIRED),
        )

    def merge_group(self, group: TensorGroup, /, **parameters: Any) -> torch.Tensor:
        base = group.base.tensor
        entries = group.non_base
        if not entries:
            return base

        # Sparsification sees independent tensors so filling later rows cannot
        # invalidate tensors saved for backward by an earlier row.
        deltas = base.new_empty((len(entries), *base.shape))
        for index, entry in enumerate(entries):
            delta = entry.tensor - base
            if self.sparsification_method:
                delta = sparsify(
                    delta,
                    density=parameters["density"][entry.id],
                    method=self.sparsification_method,
                    rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
                    **{
                        key: parameters[key][entry.id]
                        for key in ("gamma", "epsilon")
                        if key in parameters
                    },
                )
            deltas[index] = delta
            del delta

        weights = torch.tensor(
            [parameters["weight"][entry.id] for entry in entries],
            dtype=deltas.dtype,
            device=deltas.device,
        )
        while deltas.dim() > weights.dim():
            weights.unsqueeze_(-1)
        deltas.mul_(weights)

        if self.consensus_method:
            mask_dtype = torch.int8 if parameters["int8_mask"] else base.dtype
            mask = get_mask(
                deltas,
                method=self.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if parameters["normalize"]:
            mixed_delta /= divisor
        if parameters["lambda"] != 1:
            mixed_delta *= parameters["lambda"]
        return (base + mixed_delta).to(base.dtype)


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
