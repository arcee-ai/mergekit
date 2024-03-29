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

from typing import Dict, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, rectify_embed_sizes
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import MergeMethod


class ModelStockMergeTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    base_model: ModelReference
    parameter_name: str

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1 and self.base_model in tensors:
            return tensors[self.base_model]
        if len(tensors) != 3:
            raise ValueError(
                "ModelStockMerge requires exactly 3 tensors (base plus two others)"
            )

        if self.base_model not in tensors:
            raise ValueError("Base model tensor not found")
        w_0 = tensors[self.base_model]
        w_1, w_2 = [tensors[k] for k in tensors if k != self.base_model]

        rectify_embed_sizes(self.parameter_name, [w_0, w_1, w_2])

        w_1_offset = w_1 - w_0
        w_2_offset = w_2 - w_0

        norm_product = torch.norm(w_1_offset, dim=-1) * torch.norm(w_2_offset, dim=-1)
        cos_theta = (
            (w_1_offset * w_2_offset).sum(dim=-1, keepdim=True)
            / norm_product.clamp(min=1e-6)
        ).clamp(-1, 1)
        cos_theta = cos_theta.unsqueeze(-1)

        t = (2 * cos_theta) / (1 + cos_theta)

        w_12 = (w_1 + w_2) / 2
        return t * w_12 + (1 - t) * w_0


class ModelStockMerge(MergeMethod):
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        return ModelStockMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            parameter_name=output_weight.name,
        )
