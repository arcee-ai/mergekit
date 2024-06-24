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
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.metric_methods.base import MetricMethod

import torch
import torch.nn.functional as F

import numpy as np


from dataclasses import dataclass, field
from typing import Dict, List, Any


# Results
# └── layers: Dict[str, Layer]
#     └── Layer
#         ├── name: str
#         ├── metrics: Dict[str, Metric]
#         │   └── Metric
#         │       ├── name: str
#         │       ├── histogram: Histogram (optional)
#         │       │   ├── count: List[float]
#         │       │   ├── edges: List[float]
#         │       │   └── widths: List[float]
#         │       ├── mean_std: MeanStd (optional)
#         │       │   ├── mean: float
#         │       │   └── std: float (optional)
#         │       ├── heatmap: Heatmap (optional)
#         │       │   └── data: torch.Tensor
#         │       ├── value: float (optional)
#         │       └── additional_data: Dict[str, Any]
#         └── weight_info: WeightInfo

@dataclass
class MeanStd:
    mean: float
    std: Optional[float] = None

@dataclass
class Heatmap:
    data: torch.Tensor

@dataclass
class Histogram:
    count: List[float]
    edges: List[float]
    widths: List[float]

@dataclass
class Metric:
    histogram: Histogram = None
    mean_std: MeanStd = None
    heatmap: Heatmap = None
    value: float = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def filled_attributes(self) -> List[str]:
        filled_attrs = []
        for attr, value in self.__dict__.items():
            if value is not None:
                filled_attrs.append(attr)
        return filled_attrs

@dataclass
class Layer:
    metrics: Dict[str, Metric]
    weight_info: WeightInfo

    def metrics_with_property(self, prop: str) -> List[str]:
        return [name for name, metric in self.metrics.items() if getattr(metric, prop) is not None]
    
class Results:
    # Class to store the statistics for each layer, redundant - remove or add more functionality
    def __init__(self):
        self.layers: Dict[str, Layer] = {}

    def add_layer(self, layer: Layer, name: str):
        if name not in self.layers.keys():
            self.layers[name] = layer

    def get_metric(self, layer_name: str, metric_name: str) -> Metric:
        return self.get_layer(layer_name, metric_name)

def validate_tensors(tensors: List[torch.Tensor], weight_info: WeightInfo, expected_tensors: Optional[int] = 2):
    """Validate tensor shapes and count."""
    unique_shapes = set(t.shape for t in tensors)
    if len(unique_shapes) != 1:
        raise RuntimeError(f"Tensor size mismatch for {weight_info.name}, sizes: {list(unique_shapes)}")
    if expected_tensors:
        if len(tensors) != expected_tensors:
            raise RuntimeError(f"Expected {expected_tensors} tensors, got {len(tensors)}")

def ungroup_tensor(input_tensor: torch.Tensor, gqa_groups: int) -> torch.Tensor:
    """
    Ungroup a grouped tensor by repeating its rows.
    """
    rows, cols = input_tensor.shape
    new_rows = rows * gqa_groups
    ungrouped_tensor = torch.zeros(new_rows, cols)

    for i in range(gqa_groups):
        ungrouped_tensor[i*rows:(i+1)*rows] = input_tensor[i].expand(rows, -1)
    
    return ungrouped_tensor

def restructure_tensor(input_tensor: torch.Tensor, num_rows: int) -> torch.Tensor:
    """
    Restructure a tensor by splitting its rows and permuting the dimensions. 
    
    This is used so that the attention weights can be grouped by head in the first dimension.
    """
    rows, cols = input_tensor.shape
    new_rows = rows // num_rows
    reshaped_tensor = input_tensor.view(new_rows, num_rows, cols)
    restructured_tensor = reshaped_tensor.permute(1, 0, 2)
    
    return restructured_tensor

def group_attn_head_weights(k_proj: torch.Tensor, 
                            q_proj: torch.Tensor, 
                            v_proj: torch.Tensor, 
                            o_proj: torch.Tensor, 
                            weight_info: WeightInfo) -> tuple[torch.Tensor, 
                                                              torch.Tensor, 
                                                              torch.Tensor, 
                                                              torch.Tensor]:

    num_heads = weight_info.num_attention_heads
    assert num_heads is not None, "Number of attention heads is not defined"
    
    if getattr(weight_info, 'num_key_value_heads', None) and getattr(weight_info, 'num_key_value_heads', None) != 0:
        gqa_groups = num_heads // weight_info.num_key_value_heads

        k_proj = ungroup_tensor(k_proj, gqa_groups)
        v_proj = ungroup_tensor(v_proj, gqa_groups)

    k_proj = restructure_tensor(k_proj, num_heads)
    v_proj = restructure_tensor(v_proj, num_heads)
    q_proj = restructure_tensor(q_proj, num_heads)
    o_proj = restructure_tensor(o_proj.T, num_heads) # Output weights are split into heads by rows, not columns

    return k_proj, v_proj, q_proj, o_proj

def compute_histogram(tensor: torch.Tensor, n_bins: int) -> List[np.ndarray]:
    bin_counts, bin_edges = np.histogram(tensor.numpy(), bins=n_bins)
    bin_widths = np.diff(bin_edges)
    return bin_counts, bin_edges, bin_widths

def cossim_heatmap(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Normalize the rows of both matrices
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(A_norm, B_norm.t())
    
    return similarity_matrix

# Metric functions

def smape(
    tensors: List[torch.Tensor], **_kwargs
) -> Metric:
    """Symmetric Mean Absolute Percentage Error (smape)."""

    numerator = torch.abs(tensors[0] - tensors[1])
    denominator = (torch.abs(tensors[0]) + torch.abs(tensors[1]))
    smape = torch.mean(torch.div(numerator, denominator), dim=1) 
    
    hist_info = compute_histogram(smape, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=smape.mean().item(), std=smape.std().item())
    )

def cossim(
    tensors: List[torch.Tensor], return_heatmap=False, **_kwargs
) -> Metric:
    """Cosine similarity"""
    cossim = F.cosine_similarity(tensors[0], tensors[1], dim=1)

    if return_heatmap:
        heatmap = cossim_heatmap(tensors[0], tensors[1])

    assert torch.isclose(cossim, cossim, atol=1e-6).all(), "NaNs in cosine similarity"
    assert torch.isclose(cossim, cossim_heatmap(tensors[0], tensors[1]).diagonal(), atol=1e-2).all(), "Diagonal elements of cosine similarity matrix do not match"

    hist_info = compute_histogram(cossim, 100)
    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=cossim.mean().item(), std=cossim.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

def scale(
    tensors: List[torch.Tensor], return_heatmap=False, **_kwargs
) -> Metric:
    """
    Scale difference: ratio of absolute difference to average scale.
    Complementary to cosine similarity, which measures the angle between two vectors and is invariant to scale.

    values close to 0 indicate that the scales of the two vectors are similar
    """

    norm_0 = torch.norm(tensors[0], dim=1)
    norm_1 = torch.norm(tensors[1], dim=1)

    res = {}

    scale_diff = torch.abs(norm_0 - norm_1) / ((norm_0 + norm_1) / 2)

    if return_heatmap:
        norm_0 = norm_0.unsqueeze(1)  # shape becomes [num_heads, 1]
        norm_1 = norm_1.unsqueeze(0)  # shape becomes [1, num_heads]

        # Compute the scale difference between each pair of heads by broadcasting
        heatmap = torch.abs(norm_0 - norm_1) / ((norm_0 + norm_1 + 1e-10) / 2)

        assert torch.isclose(scale_diff, heatmap.diagonal(), atol=1e-4).all(), "Diagonal elements of scale difference matrix do not match"
        
    hist_info = compute_histogram(scale_diff, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=scale_diff.mean().item(), std=scale_diff.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

def mse(
    tensors: List[torch.Tensor], return_heatmap: bool =False, **_kwargs
) -> Metric:
    """Mean squared error (MSE)."""
    if return_heatmap:
        # Expand dimensions for broadcasting
        tensors_0_exp = tensors[0].unsqueeze(1)  # shape becomes [num_heads, 1, ...]
        tensors_1_exp = tensors[1].unsqueeze(0)  # shape becomes [1, num_heads, ...]

        # Compute squared differences
        diffs = (tensors_0_exp - tensors_1_exp) ** 2

        # Compute mean over all dimensions except the first two
        heatmap = diffs.mean(dim=tuple(range(2, diffs.dim()))).numpy()

    squared_diff = (tensors[0] - tensors[1]) ** 2
    mse_per_neuron = torch.mean(squared_diff, dim=1)

    hist_info = compute_histogram(mse_per_neuron, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=mse_per_neuron.mean().item(), std=mse_per_neuron.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

# Tensor Analysis (number of tensors can vary)


# Tasks

class MLPTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        weights = list(tensors.values())
        validate_tensors(weights, self.weight_info, expected_tensors=2)
        out = Layer(metrics={},
                    weight_info=self.weight_info)

        out.metrics['cossim'] = cossim(weights, return_heatmap=False)
        out.metrics['smape'] = smape(weights)
        out.metrics['scale'] = scale(weights, return_heatmap=False)
        out.metrics['mse'] = mse(weights, return_heatmap=False) # Highly inefficient

        return out

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
        self, k_proj: torch.Tensor, v_proj: torch.Tensor, q_proj: torch.Tensor, o_proj: torch.Tensor, **_kwargs
    ) -> torch.Tensor:
        # Add metrics for attention weights

        models = list(q_proj.keys())

        k_proj_0, v_proj_0, q_proj_0, o_proj_0 = group_attn_head_weights(k_proj[models[0]], q_proj[models[0]], v_proj[models[0]], o_proj[models[0]], self.weight_info)
        k_proj_1, v_proj_1, q_proj_1, o_proj_1 = group_attn_head_weights(k_proj[models[1]], q_proj[models[1]], v_proj[models[1]], o_proj[models[1]], self.weight_info)
        
        # Metrics for K, V, Q, O projections

        
        # Metrics for heads

        model_0_heads = torch.cat([k_proj_0, v_proj_0, q_proj_0, o_proj_0], dim=1)
        model_1_heads = torch.cat([k_proj_1, v_proj_1, q_proj_1, o_proj_1], dim=1)
        
        out = Layer(metrics={},
                    weight_info=self.weight_info)

        out.metrics['cossim'] = cossim([model_0_heads.view(model_0_heads.shape[0], -1), 
                                       model_1_heads.view(model_1_heads.shape[0], -1)], 
                                       return_heatmap=True)
        out.metrics['smape'] = smape([model_0_heads.view(model_0_heads.shape[0], -1),
                                        model_1_heads.view(model_1_heads.shape[0], -1)])
        out.metrics['scale'] = scale([model_0_heads.view(model_0_heads.shape[0], -1),
                                        model_1_heads.view(model_1_heads.shape[0], -1)], 
                                        return_heatmap=True)
        out.metrics['mse'] = mse([model_0_heads.view(model_0_heads.shape[0], -1),
                                    model_1_heads.view(model_1_heads.shape[0], -1)], 
                                    return_heatmap=False)

        return out

    def group_label(self) -> Optional[str]:
        return max([gather_tensor.group_label() for gather_tensor in list(self.weights.values())])
    
    def __hash__(self):
        return hash(self.weight_info)

    def __eq__(self, other):
        if not isinstance(other, AttnTask):
            return False
        return self.weight_info == other.weight_info

class DummyTask(Task[torch.Tensor]):
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

# Metric method
   
class AllMetric(MetricMethod):
    attn_weight_dict: Optional[Dict[str, torch.Tensor]] = {}
    attn_info_dict: Optional[Dict[str, WeightInfo]] = {}

    attn_parts: Optional[List[str]] = ['k_proj', 'v_proj', 'q_proj', 'o_proj'] # hard-coded for now
    block_count: Optional[int] = 0
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        parameters: Optional[Dict[str, Any]] = None,
        tensors: GatherTensors,
        **_kwargs,
    ) -> Task:
        
        if 'self_attn' in output_weight.name:
            return self.group_attn_heads(tensors, output_weight)
        elif 'mlp' in output_weight.name:        
            return MLPTask(
                gather_tensors=tensors,
                weight_info=output_weight,
                intra_model_metrics=parameters['intra_model_metrics']
            )
        else:
            # Executor expects a task to be returned
            return DummyTask(gather_tensors=tensors, weight_info=output_weight) 

        # if all attention weights are collected, create attention task
        if set(list(self.attn_weight_dict.keys())) == set(self.attn_parts):
            weights, infos = self.attn_weight_dict, self.attn_info_dict
            self.attn_weight_dict, self.attn_info_dict = {}, {}
            weight_info = WeightInfo(
                name=f"Attention Block {self.block_count}",
                force_dtype=None,
                optional=False,
                aliases=None,
                num_key_value_heads=int(infos['k_proj'].num_key_value_heads), 
                num_attention_heads=int(infos['k_proj'].num_attention_heads) 
            )
            self.block_count += 1
            return AttnTask(weights=weights, weight_infos=infos, weight_info=weight_info)
        else:
            # Executor expects a task to be returned
            return DummyTask(gather_tensors=tensors, weight_info=output_weight) 


