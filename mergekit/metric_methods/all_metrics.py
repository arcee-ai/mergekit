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

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ImmutableMap
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors, LoadTensor
from mergekit.metric_methods.base import MetricMethod
from mergekit.metric_methods.base import ConfigParameterDef

import torch
import torch.nn.functional as F

import numpy as np

def compute_histogram(tensor: torch.Tensor, n_bins: int) -> List[np.ndarray]:
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
    
    hist_info = compute_histogram(smape, 100)
    return {
        'SMAPE_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'SMAPE_mean': smape.mean().item(),
        'SMAPE_std': smape.std().item()
    }

def cossim(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:
    cossim = F.cosine_similarity(tensors[0], tensors[1], dim=1)
    if _kwargs.get('angular_distance'):
        cossim = torch.acos(cossim.clamp(min=-1, max=1))/torch.pi 

    hist_info = compute_histogram(cossim, 100)
    return {
        'cossim_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'cossim_mean': cossim.mean().item(),
        'cossim_std': cossim.std().item()
    }

def scale(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:

    norm_0 = torch.norm(tensors[0], dim=1)
    norm_1 = torch.norm(tensors[1], dim=1)

    scale_diff = torch.abs(norm_0 - norm_1) / ((norm_0 + norm_1) / 2)
        
    hist_info = compute_histogram(scale_diff, 100)
    return {
        'scale_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'scale_mean': scale_diff.mean().item(),
        'scale_std': scale_diff.std().item()
    }

def mse(
    tensors: List[torch.Tensor], **_kwargs
) -> torch.Tensor:

    squared_diff = (tensors[0] - tensors[1]) ** 2
    mse_per_neuron = torch.mean(squared_diff, dim=1)

    hist_info = compute_histogram(mse_per_neuron, 100)
    return {
        'mse_full': {
            'count': hist_info[0],
            'edges': hist_info[1],
            'widths': hist_info[2]
        },
        'mse_mean': mse_per_neuron.mean().item(),
        'mse_std': mse_per_neuron.std().item()
    }

def restructure_tensor(input_tensor, num_columns):
    """
    Restructure a tensor by splitting its columns.

    Args:
        input_tensor (torch.Tensor): The input tensor to restructure.
        num_columns (int): The number of columns for splitting.

    Returns:
        torch.Tensor: The restructured tensor.
    """
    rows, cols = input_tensor.shape
    new_cols = cols // num_columns
    reshaped_tensor = input_tensor.view(rows, num_columns, new_cols)
    restructured_tensor = reshaped_tensor.permute(1, 0, 2)
    
    return restructured_tensor

def compare_attn_head_weights(k_proj, q_proj, v_proj, o_proj, num_heads, **_kwargs):
    models = list(q_proj.keys())
    q_proj_0 = restructure_tensor(q_proj[models[0]], num_heads)
    q_proj_1 = restructure_tensor(q_proj[models[1]], num_heads)

    # Now the first dimension is the head index, so can be compared pairwise or even cross compared within/between models.
    heatmap = np.zeros((num_heads, num_heads))
    for i in range(num_heads):
        for j in range(num_heads):
            heatmap[i, j] = ((q_proj_0[i].flatten() - q_proj_1[j].flatten()) ** 2).mean().item()
    
    return {
        'MSE Attn Heatmap': heatmap,
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
        res = {}
        if 'mlp' in self.weight_info.name:

            res.update(cossim(tensors, angular_distance=self.angular_distance))
            res.update(SMAPE(tensors))
            res.update(scale(tensors))
            res.update(mse(tensors))
            
        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

class AttnTask(Task[torch.Tensor]):
    weights: Dict[str, GatherTensors]
    weight_infos: Dict[str, WeightInfo]
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:

        return self.weights

    def execute(
        self, k_proj, v_proj, q_proj, o_proj, **_kwargs
    ) -> torch.Tensor:
        # Add metrics for attention weights
        res = {}
        res.update(compare_attn_head_weights(k_proj, q_proj, v_proj, o_proj, num_heads=32)) # 32 is a placeholder

        return res

    def group_label(self) -> Optional[str]:
        # Use max of the group labels
        return max([gather_tensor.group_label() for gather_tensor in list(self.weights.values())]) # Check this (X)
    
    def __hash__(self):
        return hash((tuple(self.weight_infos),))

    def __eq__(self, other):
        if not isinstance(other, AttnTask):
            return False
        return self.weight_infos == other.weight_infos

class blankTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo
    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, **_kwargs
    ) -> torch.Tensor:
        
        return 
    
    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()
        

class AllMetric(MetricMethod):
    attn_weight_tensors: Optional[list] = []
    attn_weight_infos: Optional[list] = []

    attn_weight_dict: Optional[Dict[str, torch.Tensor]] = {}
    attn_info_dict: Optional[Dict[str, WeightInfo]] = {}


    attn_parts: Optional[List[str]] = ['k_proj', 'v_proj', 'q_proj', 'o_proj']
    block_count: Optional[int] = 0
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
        
        if 'self_attn' in output_weight.name:
            for part in self.attn_parts: # also check only one key
                if part in output_weight.name:
                   self.attn_weight_dict[part] = tensors
                   self.attn_info_dict[part] = output_weight   

            if set(list(self.attn_weight_dict.keys())) == set(self.attn_parts):
                weights = self.attn_weight_dict
                infos = self.attn_info_dict
                self.attn_weight_dict = {}
                self.attn_info_dict = {}
                weight_info = WeightInfo(
                    name=f"Attention Block {self.block_count}",
                    force_dtype=None,
                    optional=False,
                    aliases=None,
                )
                self.block_count += 1
                return AttnTask(weights=weights, weight_infos=infos, weight_info=weight_info)
        if 'mlp' in output_weight.name:        
            return AllMetricTask(
                gather_tensors=tensors,
                weight_info=output_weight,
                angular_distance=parameters["angular_distance"]
            )
        else:
            return blankTask(gather_tensors=tensors, weight_info=output_weight)


