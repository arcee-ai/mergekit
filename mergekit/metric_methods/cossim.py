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

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.metric_methods.base import MetricMethod

import torch.nn.functional as F

class CossimMetricTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]

        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )
        if len(tensors) != 2:
            raise RuntimeError(f"Expected 2 tensors, got {len(tensors)}")

        if 'mlp' not in self.weight_info.name:
            return

        res = {}
        # pairwise similarity of corresponding rows in weights matrix

        res['cossim_full'] = F.cosine_similarity(tensors[0], tensors[1], dim=1) # this might get memory intensive, consider binning
        res['cossim_mean'] = res['cossim_full'].mean()
        res['cossim_std'] = res['cossim_full'].std()

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class CossimMetric(MetricMethod):
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        **_kwargs,
    ) -> Task:
        return CossimMetricTask(
            gather_tensors=tensors,
            weight_info=output_weight,
        )
