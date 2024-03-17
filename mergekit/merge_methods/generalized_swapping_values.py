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
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.sparsify import SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedSwappingValues(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
            ConfigParameterDef(name="diagonal_offset", required=True),
            ConfigParameterDef(name="invert_offset", required=False, default_value= False),
            ConfigParameterDef(name="random_mask", required=False, default_value= 0.0),
            ConfigParameterDef(name="random_mask_seed", required=False, default_value= None),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        return GTATask(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            out_tensor_name=output_weight.name,
        )


class GTATask(Task[torch.Tensor]):
    method: GeneralizedSwappingValues
    tensors: GatherTensors
    base_model: ModelReference
    out_tensor_name: str
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.out_tensor_name,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
        )
        if not tvs:
            return base

        # sparsify
        if self.method.sparsification_method:
            for tv_info in tvs:
                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                )

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if self.normalize:
            mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)

def swapping(base, x, parameters, parameter_name):
    def swap_values(shape, n, base, x):
        if x.dim() == 2:
           rows, cols = shape
           rows_range = torch.arange(rows).view(-1, 1)
           cols_range = torch.arange(cols).view(1, -1)
           mask = ((rows_range + cols_range) % n == 0).bool()
           x = torch.where(mask, x, base)
        else:
           rows_range = torch.arange(shape[0])
           mask = ((rows_range) % n == 0).bool()
           x = torch.where(mask, x, base)
        return x

    def rand_mask(base, x, percent, seed=None):
        oldseed = torch.seed()
        if seed is not None:
            torch.manual_seed(seed)
        random = torch.rand(base.shape)
        mask = random <= percent
        del random
        torch.manual_seed(oldseed)
        x = torch.where(mask, x, base) 
        return x
    
    bt = base.dtype
    if x.device.type == "cpu":
        x = x.to(torch.float32)
        base = base.to(torch.float32)

    diagonal_offset = None
    diagonal_offset = parameters.get('diagonal_offset')
    random_mask = parameters.get('random_mask')
    random_mask_seed = parameters.get('random_mask_seed')
    random_mask_seed = int(random_mask_seed) if random_mask_seed is not None else random_mask_seed

    assert (diagonal_offset is not None) and (diagonal_offset % 1 == 0) and (diagonal_offset >= 2), "The diagonal_offset must be an integer greater than or equal to 2."
        
    if random_mask != 0.0:
       assert (random_mask is not None) and (random_mask < 1.0) and (random_mask > 0.0) , "The random_mask parameter can't be empty, 0, 1, or None, it must be a number between 0 and 1."
       assert random_mask_seed is None or (isinstance(random_mask_seed, int) and random_mask_seed % 1 == 0), "The random_mask_seed parameter must be None or an integer, None is a random seed."
       x = rand_mask(base, x, random_mask, random_mask_seed)

    else:
       if parameters.get('invert_offset') == False:
           x = swap_values(x.shape, diagonal_offset, base, x)
       else:
           x = swap_values(x.shape, diagonal_offset, x, base)

    del base
    return x.to(bt)

def get_task_vectors(
    parameter_name: str,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    res = []
    for model in keys:
        if model == base_model:
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

        x = swapping(base, x, dict(tensor_parameters[model].items()), parameter_name)

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base


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
