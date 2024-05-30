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

class ScaleMetricTask(Task[torch.Tensor]):
    """
    Relative difference in scale per neuron. Complemetary to the cosine similarity metric.

    scale_diff (X, Y) = absolute value of the difference in magnitude of X and Y, normalized by the average magnitude of X and Y
    """
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

        # Ensure the tensors have the same shape
        assert tensors[0].shape == tensors[0].shape, "Tensors must have the same shape"
        
        # 
        norm_0 = torch.norm(tensors[0], dim=1)
        norm_1 = torch.norm(tensors[1], dim=1)

        scale_diff = torch.abs(norm_0 - norm_1)
        scale_diff = scale_diff / ((norm_0 + norm_1) / 2)
         
        res['scale_full'] = scale_diff
        res['scale_mean'] = scale_diff.mean()
        res['scale_std'] = scale_diff.std()

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class ScaleMetric(MetricMethod):
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        **_kwargs,
    ) -> Task:
        return ScaleMetricTask(
            gather_tensors=tensors,
            weight_info=output_weight,
        )
