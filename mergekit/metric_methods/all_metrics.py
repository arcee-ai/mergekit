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
from mergekit.metric_methods.base import MetricMethod, Layer
import torch
from typing import Dict, List, Any
from mergekit.metric_methods.metrics import cosine_similarity, smape, scale, mse, weight_magnitude, numerical_rank

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
    
    return ungrouped_tensor.to(input_tensor.device)

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
        layer_results = Layer(metrics={},
                    weight_info=self.weight_info)

        if self.intra_model_metrics:
            validate_tensors(weights, self.weight_info, expected_tensors=1)
            layer_results.add_metric(weight_magnitude(weights[0]), name='weight_magnitude')
            layer_results.add_metric(numerical_rank(weights[0]), name='numerical_rank')
        else:
            validate_tensors(weights, self.weight_info, expected_tensors=2)
            layer_results.add_metric(cosine_similarity(weights, return_heatmap=False), name = 'cosine_similarity')
            layer_results.add_metric(smape(weights), name = 'smape')
            layer_results.add_metric(scale(weights, return_heatmap=False), name = 'scale')
            layer_results.add_metric(mse(weights, return_heatmap=False), name = 'mse')


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
        model_0_heads = torch.cat([k_proj_0, v_proj_0, q_proj_0, o_proj_0], dim=1)
        layer_results = Layer(metrics={},
                    weight_info=self.weight_info)
        
        
        if self.intra_model_metrics:
        
            layer_results.add_metric(
                metric=weight_magnitude(model_0_heads),
                name='weight_magnitude'
                )
        else:
        
            k_proj_1, v_proj_1, q_proj_1, o_proj_1 = group_attn_head_weights(k_proj[model_references[1]], 
                                                                            q_proj[model_references[1]], 
                                                                            v_proj[model_references[1]], 
                                                                            o_proj[model_references[1]], 
                                                                            self.weight_info)
            

            model_1_heads = torch.cat([k_proj_1, v_proj_1, q_proj_1, o_proj_1], dim=1)
        


            layer_results.add_metric(cosine_similarity([model_0_heads.view(model_0_heads.shape[0], -1),
                                    model_1_heads.view(model_1_heads.shape[0], -1)],
                                    return_heatmap=True), 
                                    name = 'cosine_similarity')
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
    intra_model_metrics: bool = False

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        
        tensors = list(tensors.values())
        
        assert tensors[0].dim() == 1, "LayerNorm tensors must be 2D"

        layer_results = Layer(metrics={}, weight_info=self.weight_info)

        if self.intra_model_metrics:
            layer_results.add_metric(
                metric=weight_magnitude(tensors[0].unsqueeze(1)),
                name='weight_magnitude'
                )
        else:
            assert tensors[1].dim() == 1, "LayerNorm tensors must be 2D"
            layer_results.add_metric(cosine_similarity([tensors[0].unsqueeze(1), tensors[1].unsqueeze(1)], return_heatmap=True), name = 'cosine_similarity')
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
    def __init__(self) -> None:
        super().__init__()
        self.attn_weight_dict: Optional[Dict[str, torch.Tensor]] = {}
        self.attn_info_dict: Optional[Dict[str, WeightInfo]] = {}
        self.attn_parts: Optional[List[str]] = ['k_proj', 'v_proj', 'q_proj', 'o_proj'] # hard-coded for now
        self.block_count: Optional[int] = 0

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
            return LayerNormTask(gather_tensors=tensors, weight_info=output_weight, intra_model_metrics=parameters['intra_model_metrics'])
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


