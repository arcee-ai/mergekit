# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod
from mergekit.sparsify import SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool

    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            parameter_name,
            config,
            input_tensors,
            required_parameters=["weight"],
            optional_parameters=["density"],
        )
        if not tvs:
            return base

        # sparsify
        if self.sparsification_method:
            for tv_info in tvs:
                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info.get("density", 1.0),
                    method=self.sparsification_method,
                )

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.consensus_method:
            mask_dtype = (
                torch.int8
                if config.parameter("int8_mask", default=False)
                else base.dtype
            )
            mask = get_mask(
                weighted_deltas, method=self.consensus_method, mask_dtype=mask_dtype
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if config.parameter("normalize", default=self.default_normalize):
            mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)


def get_task_vectors(
    parameter_name: str,
    config: ConfigReader,
    input_tensors: Dict[TensorReference, torch.Tensor],
    required_parameters: Optional[List[str]] = None,
    optional_parameters: Optional[List[str]] = None,
) -> Tuple[List[torch.Tensor], List[float], torch.Tensor]:
    tensors = {tr.model: value for (tr, value) in input_tensors.items()}
    keys = list(tensors.keys())
    base = tensors[config.base_model]

    res = []
    for model in keys:
        if model == config.base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if "lm_head" in parameter_name or "embed_tokens" in parameter_name:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in required_parameters:
            d[p] = config.parameter(p, model, required=True)
        for p in optional_parameters:
            d[p] = config.parameter(p, model, required=False)
        res.append(d)
    return res, base


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
