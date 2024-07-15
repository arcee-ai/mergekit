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
from typing import Any, Dict, List, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class ModelStockMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    filter_wise: bool = False

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1 and self.base_model in tensors:
            return tensors[self.base_model]
        if len(tensors) < 3:
            if self.weight_info.optional:
                logging.warning(
                    f"Optional weight {self.weight_info.name} not present in enough models, discarding"
                )
                return None

            raise ValueError(
                "ModelStockMerge requires at least 3 models (base plus two+ others)"
            )

        w_0, ws = self.get_rectified_weights(tensors)
        out_shape = w_0.shape

        if self.filter_wise:
            if w_0.dim() == 1:
                # bias (or other single-vector) parameters should be treated as row vectors
                w_0 = w_0.unsqueeze(0)
                ws = [w.unsqueeze(0) for w in ws]
        else:
            w_0 = w_0.view(-1)
            ws = [w.view(-1) for w in ws]

        offsets = [w - w_0 for w in ws]

        # now there is a question of how to come up with a value for theta.
        # in the two-vector case, we can get an exact angle between the two vectors
        # but the paper doesn't explicitly say what to do in the multi-vector case -
        # they keep using a singular theta value and don't elaborate on how to
        # calculate it. i'm going to assume an average of pairwise angles for now? i guess?

        cos_thetas = []
        for i, w_0_offset in enumerate(offsets):
            for j in range(i + 1, len(offsets)):
                w_1_offset = offsets[j]

                norm_product = torch.norm(w_0_offset, dim=-1) * torch.norm(
                    w_1_offset, dim=-1
                )
                cos_theta = (
                    (w_0_offset * w_1_offset).sum(dim=-1) / norm_product.clamp(min=1e-6)
                ).clamp(-1, 1)
                cos_thetas.append(cos_theta)

        cos_theta = torch.stack(cos_thetas).mean(dim=0).unsqueeze(-1)
        N = len(ws)
        t = (N * cos_theta) / (1 + (N - 1) * cos_theta)

        w_avg = sum(ws) / len(ws)
        w_h = t * w_avg + (1 - t) * w_0

        return w_h.reshape(out_shape)

    def get_rectified_weights(self, tensors: Dict[ModelReference, torch.Tensor]):
        if self.base_model not in tensors:
            raise ValueError("Base model tensor not found")

        all_weights = [tensors[self.base_model]] + [
            tensors[k] for k in tensors if k != self.base_model
        ]
        rectify_embed_sizes(self.weight_info, all_weights)
        w_0 = all_weights[0]
        ws = all_weights[1:]
        return w_0, ws

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class ModelStockMerge(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="filter_wise", required=False, default_value=False)
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        **_kwargs,
    ) -> Task:
        return ModelStockMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            filter_wise=parameters["filter_wise"],
        )
