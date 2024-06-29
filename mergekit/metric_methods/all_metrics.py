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

from typing import Any, Dict, List, Optional, Union

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.metric_methods.base import MetricMethod, MeanStd, Heatmap, Histogram, Metric, Layer
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any

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

# Tensor Comparisons (Require exactly 2 tensors)

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

def weight_magnitude(tensors: List[torch.Tensor], model_refs: List[ModelReference]) -> List[Metric]:
    output = []
    for tensor, model_reference in zip(tensors, model_refs):
        weight_magnitudes = torch.abs(tensor.flatten())
        hist_info = compute_histogram(weight_magnitudes, 100)
        output.append(Metric(
            histogram=Histogram(count=hist_info[0], 
                                edges=hist_info[1], 
                                widths=hist_info[2]
                                ),
            mean_std=MeanStd(mean=weight_magnitudes.mean().item(),
                                std=weight_magnitudes.std().item()),
            model_ref=model_reference
            ))
    return output

def numerical_rank(tensors: List[torch.Tensor], model_refs: List[ModelReference], epsilon: float = 1e-5) -> List[Metric]:
    """
    Computes the numerical rank of the representations matrix X based on the singular values
    of its sample covariance matrix. The rank is determined as the number of singular values
    above a threshold. The threshold is defined as the highest singular value times a given epsilon.

    Parameters:
    - X : torch.Tensor
        The representations matrix from which the sample covariance matrix will be computed.
    - epsilon : float, optional
        The factor to multiply with the highest singular value to set the threshold (default is 1e-3).
    - flip : bool, optional - allows transpose for efficient computation. False only used in testing
    Returns:
    - int
        The numerical rank of the matrix.

    Implemented according to description in the paper:
        The Tunnel Effect: Building Data Representations in Deep Neural Networks
        https://arxiv.org/pdf/2305.19753.pdf

    """
    output = []
    for tensor, model_reference in zip(tensors, model_refs):
        
        # Center the data by subtracting the mean
        X_centered = tensor - torch.mean(tensor, dim=0)
        X_std = torch.std(X_centered, dim=0, unbiased=False)
        X_centered /= X_std

        # Compute the sample covariance matrix
        covariance_matrix = X_centered.t() @ X_centered / (tensor.shape[0] - 1)
        # Compute singular values using SVD on the covariance matrix
        U, singular_values, V = torch.svd(covariance_matrix)
        # Determine the threshold
        threshold = singular_values[0] * epsilon
        # Count singular values greater than the threshold
        num_rank = torch.sum(singular_values > threshold).item()

        value = int(num_rank)

        output.append(
            Metric(
                model_ref=model_reference,
                mean_std=MeanStd(
                    mean=value, 
                    std=None), 
                ))

    return output

# Tasks

class MLPTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo
    intra_model_metrics: bool = False

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        weights = list(tensors.values())
        validate_tensors(weights, self.weight_info, expected_tensors=2)
        layer_results = Layer(metrics={},
                    weight_info=self.weight_info)

        layer_results.add_metric(cossim(weights, return_heatmap=False), name = 'cossim')
        layer_results.add_metric(smape(weights), name = 'smape')
        layer_results.add_metric(scale(weights, return_heatmap=False), name = 'scale')
        layer_results.add_metric(mse(weights, return_heatmap=False), name = 'mse')

        if self.intra_model_metrics:
            model_refs = list(tensors.keys())
            layer_results.add_metric_list(metric_list=weight_magnitude(weights, model_refs), name='weight_magnitude')
            layer_results.add_metric_list(metric_list=numerical_rank(weights, model_refs), name='numerical_rank')

        return layer_results

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

class AttnTask(Task[torch.Tensor]):
    weights: Dict[str, GatherTensors]
    weight_infos: Dict[str, WeightInfo]
    weight_info: WeightInfo
    intra_model_metrics: bool = False

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:

        return self.weights

    def execute(
        self, k_proj: torch.Tensor, v_proj: torch.Tensor, q_proj: torch.Tensor, o_proj: torch.Tensor, **_kwargs
    ) -> torch.Tensor:
        # Add metrics for attention weights

        model_references = list(q_proj.keys())

        k_proj_0, v_proj_0, q_proj_0, o_proj_0 = group_attn_head_weights(k_proj[model_references[0]], 
                                                                         q_proj[model_references[0]], 
                                                                         v_proj[model_references[0]], 
                                                                         o_proj[model_references[0]], 
                                                                         self.weight_info)
        k_proj_1, v_proj_1, q_proj_1, o_proj_1 = group_attn_head_weights(k_proj[model_references[1]], 
                                                                         q_proj[model_references[1]], 
                                                                         v_proj[model_references[1]], 
                                                                         o_proj[model_references[1]], 
                                                                         self.weight_info)
        
        # Metrics for K, V, Q, O projections

        
        # Metrics for heads

        model_0_heads = torch.cat([k_proj_0, v_proj_0, q_proj_0, o_proj_0], dim=1)
        model_1_heads = torch.cat([k_proj_1, v_proj_1, q_proj_1, o_proj_1], dim=1)
        
        layer_results = Layer(metrics={},
                    weight_info=self.weight_info)


        layer_results.add_metric(cossim([model_0_heads.view(model_0_heads.shape[0], -1),
                                model_1_heads.view(model_1_heads.shape[0], -1)],
                                return_heatmap=True), 
                                name = 'cossim')
        layer_results.add_metric(smape([model_0_heads.view(model_0_heads.shape[0], -1),
                                model_1_heads.view(model_1_heads.shape[0], -1)]), 
                                name = 'smape')
        layer_results.add_metric(scale([model_0_heads.view(model_0_heads.shape[0], -1),
                                model_1_heads.view(model_1_heads.shape[0], -1)], 
                                return_heatmap=True), 
                                name = 'scale')
        layer_results.add_metric(mse([model_0_heads.view(model_0_heads.shape[0], -1),
                            model_1_heads.view(model_1_heads.shape[0], -1)], 
                            return_heatmap=False), 
                            name = 'mse')
        
        if self.intra_model_metrics:
        
            layer_results.add_metric_list(
                metric_list=weight_magnitude([model_0_heads, model_1_heads], model_refs=model_references),
                name='weight_magnitude'
                )
            

        return layer_results

    def group_label(self) -> Optional[str]:
        return max([gather_tensor.group_label() for gather_tensor in list(self.weights.values())])
    
    def __hash__(self):
        return hash(self.weight_info)

    def __eq__(self, other):
        if not isinstance(other, AttnTask):
            return False
        return self.weight_info == other.weight_info

class LayerNormTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo
    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        
        tensors = list(tensors.values())
        
        assert tensors[0].dim() == 1, "LayerNorm tensors must be 2D"
        assert tensors[1].dim() == 1, "LayerNorm tensors must be 2D"

        layer_results = Layer(metrics={}, weight_info=self.weight_info)
        
        layer_results.add_metric(cossim([tensors[0].unsqueeze(1), tensors[1].unsqueeze(1)], return_heatmap=True), name = 'cossim')
        layer_results.add_metric(smape([tensors[0].unsqueeze(1), tensors[1].unsqueeze(1)]), name = 'smape')
        layer_results.add_metric(scale([tensors[0].unsqueeze(1), tensors[1].unsqueeze(1)], return_heatmap=True), name = 'scale')
        layer_results.add_metric(mse([tensors[0].unsqueeze(1), tensors[1].unsqueeze(1)], return_heatmap=True), name = 'mse')

        return layer_results
    
    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

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


from mergekit.merge_methods.base import ConfigParameterDef

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
            return self.group_attn_heads(tensors, output_weight, parameters)
        elif 'mlp' in output_weight.name:        
            return MLPTask(
                gather_tensors=tensors,
                weight_info=output_weight,
                intra_model_metrics=parameters['intra_model_metrics']
            )
        elif 'layernorm' in output_weight.name:
            return LayerNormTask(gather_tensors=tensors, weight_info=output_weight)
        else:
            # Executor expects a task to be returned
            return DummyTask(gather_tensors=tensors, weight_info=output_weight) 
        
    def group_attn_heads(self, tensors: GatherTensors, output_weight: WeightInfo, parameters: Dict[str, Any]):
        # collect all attention weights
        for part in self.attn_parts: # also check only one key
            if part in output_weight.name:
                assert self.attn_weight_dict.get(part) is None, f"Duplicate attention part {part}"
                self.attn_weight_dict[part] = tensors
                self.attn_info_dict[part] = output_weight   

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
            return AttnTask(weights=weights, weight_infos=infos, weight_info=weight_info, intra_model_metrics=parameters['intra_model_metrics'])
        else:
            # Executor expects a task to be returned
            return DummyTask(gather_tensors=tensors, weight_info=output_weight) 
        

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="intra_model_metrics", required=False, default_value=False),
        ]


