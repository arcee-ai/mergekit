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
from mergekit.metric_methods.base import ConfigParameterDef
import torch.nn.functional as F
import numpy as np

def binning(tensor: torch.Tensor, n_bins: int) -> List[np.ndarray]:
    bin_counts, bin_edges = np.histogram(tensor.numpy(), bins=n_bins)
    bin_widths = np.diff(bin_edges)
    return bin_counts, bin_edges, bin_widths

def validate_tensors(tensors: List[torch.Tensor], weight_info: WeightInfo, expected_tensors: Optional[int] = 2):
    unique_shapes = set(t.shape for t in tensors)
    if len(unique_shapes) != 1:
        raise RuntimeError(f"Tensor size mismatch for {weight_info.name}, sizes: {list(unique_shapes)}")
    if expected_tensors:
        if len(tensors) != expected_tensors:
            raise RuntimeError(f"Expected {expected_tensors} tensors, got {len(tensors)}")
    
def SMAPE(
    tensors: List[torch.Tensor], **_kwargs
) -> Dict[str, Any]:
    numerator = torch.abs(tensors[0] - tensors[1])
    denominator = (torch.abs(tensors[0]) + torch.abs(tensors[1]))
    smape = torch.mean(torch.div(numerator, denominator), dim=1) 
    
    hist_info = binning(smape, 100)
    return {
        'SMAPE_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'SMAPE_mean': smape.mean(),
        'SMAPE_std': smape.std()
    }

def cossim(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:
    cossim = F.cosine_similarity(tensors[0], tensors[1], dim=1)
    if _kwargs.get('angular_distance'):
        cossim = torch.acos(cossim.clamp(min=-1, max=1))/torch.pi 

    hist_info = binning(cossim, 100)
    return {
        'cossim_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'cossim_mean': cossim.mean(),
        'cossim_std': cossim.std()
    }

def scale(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:

    norm_0 = torch.norm(tensors[0], dim=1)
    norm_1 = torch.norm(tensors[1], dim=1)

    scale_diff = torch.abs(norm_0 - norm_1)
    scale_diff = scale_diff / ((norm_0 + norm_1) / 2)
        
    hist_info = binning(scale_diff, 100)
    return {
        'scale_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'scale_mean': scale_diff.mean(),
        'scale_std': scale_diff.std()
    }

def mse(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:
    # Compute the squared differences
    squared_diff = (tensors[0] - tensors[1]) ** 2
    
    
    # Compute the mean of squared differences for each row
    mse_per_neuron = torch.mean(squared_diff, dim=1)

    hist_info = binning(mse_per_neuron, 100)
    return {
        'mse_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'mse_mean': mse_per_neuron.mean(),
        'mse_std': mse_per_neuron.std()
    }

class AllMetricTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo
    angular_distance: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        tensors = list(tensors.values())
        validate_tensors(tensors, self.weight_info, expected_tensors=2)
        if 'mlp' not in self.weight_info.name:
            return

        res = {}

        res.update(cossim(tensors, angular_distance=self.angular_distance))
        res.update(SMAPE(tensors))
        res.update(scale(tensors))
        res.update(mse(tensors))
            
        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class AllMetric(MetricMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="angular_distance", required=False, default_value=False),
        ]
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        parameters: Optional[Dict[str, Any]] = None,
        tensors: GatherTensors,
        **_kwargs,
    ) -> Task:
        return AllMetricTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            angular_distance=parameters["angular_distance"]
        )


