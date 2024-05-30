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

class SMAPEMetricTask(Task[torch.Tensor]):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE = 100 * |y - y_hat| / ((|y| + |y_hat|) / 2)
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
        
        # SMAPE
        numerator = torch.abs(tensors[0] - tensors[1])
        denominator = (torch.abs(tensors[0]) + torch.abs(tensors[1])) / 2
        smape = 100 * torch.mean(torch.div(numerator, denominator), dim=1) 
        
        res['SMAPE_full'] = smape
        res['SMAPE_mean'] = smape.mean()
        res['SMAPE_std'] = smape.std()

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class SMAPEMetric(MetricMethod):
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        **_kwargs,
    ) -> Task:
        return SMAPEMetricTask(
            gather_tensors=tensors,
            weight_info=output_weight,
        )
