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
import torch.nn.functional as F
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.generalized_task_arithmetic import get_task_vectors


class PCBMerge(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", required=True),
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        **kwargs,
    ) -> Task[torch.Tensor]:
        return PCBMergeTask(
            output_weight=output_weight,
            tensors=tensors,
            base_model=base_model,
            density=parameters["density"],
            weight=parameters["weight"],
        )


class PCBMergeTask(Task[torch.Tensor]):
    output_weight: WeightInfo
    tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    density: float
    weight: float

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
        tv_info, base = get_task_vectors(
            self.output_weight,
            self.base_model,
            tensors,
            tensor_parameters=ImmutableMap({model: {} for model in tensors}),
        )
        if not tv_info:
            return base

        n = len(tv_info)
        tvs = torch.stack([tv["delta"] for tv in tv_info], dim=0)
        tvs_flat = tvs.view(n, -1)

        # $b_i = b_{intra, i} \odot b_{inter, i}$
        # $b_{intra, i} = Softmax(N \cdot Norm(\delta_i \odot \delta_i))$
        norm_tvs_sqr = F.normalize(tvs_flat * tvs_flat, dim=1)
        b_intra = F.softmax(n * norm_tvs_sqr, dim=1)

        # $b_{inter, i} = \sum_{j = 1}^{n} Softmax(Norm(\delta_i \odot \delta_j))$
        b_inter = torch.zeros_like(tvs_flat)
        for i in range(n):
            inter_prod = tvs_flat[i] * tvs_flat
            inter_norm = F.normalize(inter_prod, dim=1)
            b_inter[i] = F.softmax(inter_norm, dim=1).sum(dim=0)

        b = b_intra * b_inter
        k = int(tvs_flat.shape[1] * self.density)
        # $m_i = b_i \geq sorted(b_i)[k]$
        # threshold = torch.kthvalue(b, k).values
        # m = (b >= threshold.unsqueeze(1)).float()
        _, indices = torch.topk(b, k, dim=1)
        m = torch.zeros_like(b)
        m.scatter_(1, indices, 1)

        # $\hat{b}_i = b_i \odot m_i$
        b_hat = b * m

        weights = b_hat / torch.sum(b_hat)
        final_delta = torch.sum(tvs_flat * weights, dim=0).view(tvs.shape[1:])
        return base + self.weight * final_delta
