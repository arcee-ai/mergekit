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

from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel
from torch._tensor import Tensor

from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.merge_methods.slerp import slerp
from mergekit.tokenizer import BuildTokenizer, TokenizerInfo


class TokenizerPermutationMergeTask(Task[torch.Tensor]):
    tokenizer_task: BuildTokenizer
    gather_tensors: GatherTensors
    base_model: Optional[ModelReference]
    use_slerp: bool
    slerp_t: float
    tensor_parameters: ImmutableMap[ModelReference, Any]

    def arguments(self) -> Dict[str, Task]:
        return {"tokenizer_info": self.tokenizer_task, "tensors": self.gather_tensors}

    def execute(
        self, tokenizer_info: TokenizerInfo, tensors: Dict[ModelReference, torch.Tensor]
    ) -> Tensor:
        if not tensors:
            return None
        if len(tensors) == 1:
            return list(tensors.values())[0]

        models = []
        expanded = []
        masks = []
        weights = []
        for model in tensors:
            models.append(model)

            x = tensors[model]
            p = tokenizer_info.permutations[model].to(dtype=x.dtype, device=x.device)
            temp_dtype = torch.float32 if x.device.type == "cpu" else x.dtype
            if p.shape[1] == x.shape[0]:
                xp = (p.to(dtype=temp_dtype) @ x.to(dtype=temp_dtype)).to(x.dtype)
            else:
                raise RuntimeError("Shape mismatch")

            expanded.append(xp)
            masks.append(p.sum(dim=-1, keepdim=True) > 0)

            is_base = model == self.base_model
            if self.use_slerp:
                weight = (1.0 - self.slerp_t) if is_base else self.slerp_t
            else:
                weight = self.tensor_parameters[model]["weight"]

            weights.append(weight)

        expanded = torch.stack(expanded, dim=0)
        masks = torch.stack(masks, dim=0)
        weights = (
            torch.tensor(weights, dtype=expanded.dtype, device=expanded.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        total_weight = (masks * weights).sum(dim=0)
        scale = 1 / total_weight
        scale[total_weight.abs() < 1e-8] = 0

        linear_merged = (expanded * weights * masks).sum(dim=0) * scale

        if self.use_slerp:
            if expanded.shape[0] != 2:
                raise RuntimeError("SLERP takes exactly two models")

            if models[0] == self.base_model:
                v0 = expanded[0, ...]
                v1 = expanded[1, ...]
            else:
                v0 = expanded[1, ...]
                v1 = expanded[0, ...]

            res = slerp(self.slerp_t, v0, v1)
            need_linear = (masks.sum(dim=0) != 2).squeeze(dim=-1)
            res[need_linear, :] = linear_merged[need_linear, :].to(
                device=res.device, dtype=res.dtype
            )
            return res

        return linear_merged


class TokenizerPermutationMerge(MergeMethod, BaseModel):
    tokenizer_task: BuildTokenizer

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("t", required=False),
            ConfigParameterDef(name="embed_slerp", required=False, default_value=False),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("weight", required=False),
        ]

    def make_task(
        self,
        *,
        tensors: GatherTensors,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        return TokenizerPermutationMergeTask(
            base_model=base_model,
            tokenizer_task=self.tokenizer_task,
            gather_tensors=tensors,
            use_slerp=parameters["embed_slerp"],
            slerp_t=parameters["t"],
            tensor_parameters=tensor_parameters,
        )
