# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
from typing_extensions import Literal, override

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    InputParameterTarget,
    MergeMethod,
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
class GeneralizedTaskArithmeticMerge(MergeMethod):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    def name(self) -> str:
        return self.method_name

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name

    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url

    @property
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
                ParameterScope.INPUT,
                input_target=InputParameterTarget.NON_BASE,
            ),
            ParameterSpec(
                "density",
                float,
                ParameterScope.INPUT,
                input_target=InputParameterTarget.NON_BASE,
                default=1.0,
            ),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            params.append(
                ParameterSpec(
                    "gamma",
                    float,
                    ParameterScope.INPUT,
                    input_target=InputParameterTarget.NON_BASE,
                    default=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.della_magprune:
            params.append(
                ParameterSpec(
                    "epsilon",
                    float,
                    ParameterScope.INPUT,
                    input_target=InputParameterTarget.NON_BASE,
                    default=0.15,
                )
            )
        return MergeMethodSpec(
            name=self.method_name,
            pretty_name=self.method_pretty_name,
            reference_url=self.method_reference_url,
            parameters=tuple(params),
            contract=InputContract(
                base=BasePolicy.REQUIRED,
                min_inputs=1,
                min_non_base=0,
            ),
        )

    def merge(self, group: TensorGroup, **parameters: Any) -> torch.Tensor:
        base = group.base.tensor
        task_vectors = []
        for entry in group.non_base:
            tensor = entry.tensor.to(base.dtype)
            if tensor.shape != base.shape:
                if group.metadata.is_embed:
                    tensor = tensor[: base.shape[0], : base.shape[1]]
                    logging.warning(
                        f"Using submatrix of {entry.id}:{group.metadata.name}"
                    )
                else:
                    logging.warning(
                        f"Skipping {entry.id}:{group.metadata.name} due to size mismatch"
                    )
                    continue
            info = {
                "delta": tensor - base,
                "weight": parameters["weight"][entry.id],
                "density": parameters["density"][entry.id],
            }
            for optional_name in ("gamma", "epsilon"):
                if optional_name in parameters:
                    info[optional_name] = parameters[optional_name][entry.id]
            task_vectors.append(info)

        if not task_vectors:
            return base

        rescale_norm = RescaleNorm.l1 if parameters["rescale"] else None
        if self.sparsification_method:
            for info in task_vectors:
                kwargs = {key: info[key] for key in ("gamma", "epsilon") if key in info}
                info["delta"] = sparsify(
                    info["delta"],
                    density=info["density"],
                    method=self.sparsification_method,
                    rescale_norm=rescale_norm,
                    **kwargs,
                )

        deltas = torch.stack([info["delta"] for info in task_vectors], dim=0)
        weights = torch.tensor(
            [info["weight"] for info in task_vectors],
            dtype=deltas.dtype,
            device=deltas.device,
        )
        while deltas.dim() > weights.dim():
            weights.unsqueeze_(-1)
        weighted_deltas = deltas * weights

        if self.consensus_method:
            mask_dtype = torch.int8 if parameters["int8_mask"] else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
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
