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

from typing import Any, Dict, List, Tuple

import torch
from torch._tensor import Tensor
from mergekit.graph import Task

class ZipMergeUnMergeMatrixTask(Task[torch.Tensor]):
    covariance_tensor: Task

    def arguments(self) -> Dict[str, Task]:
        return {"covariance_tensor": self.covariance_tensor}

    def execute(
        self,
        covariance_tensor: torch.Tensor,
    ) -> Tuple[Tensor, Tensor]:

        prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
        # skip metrics associated with residuals and qk if qk is true
        correlation_matrix = None
        if prev_node_layer == None or not contains_name(prev_node_layer,special_cases_nodes):
            if node in corrs:
                correlation_matrix = corrs[node]

            info = self.graphs[0].get_node_info(node)
            print(info)
            next_node_info = self.graphs[0].get_node_info(node+1)['layer']

            # Handle Attention Merging
            if next_node_info != None and (self.graphs[0].modules['lin_attn'] in next_node_info):
                layer_no = [int(i) for i in self.graphs[0].get_node_info(node+1)['layer'].split('.') if i.isdigit()][0]
                if transform_fn.__name__ in ['match_tensors_permute'] and ignore_heads == False:
                    n_heads = self.graphs[0].num_heads
                    mha_transform_fn = transform_fn.__name__ + '_MHA' 
                    merge, unmerge, attn_head_perm, cost = get_merging_fn(mha_transform_fn)(n_heads, r=reduce_ratio, 
                                                                                        permute_heads=permute_heads, print_costs=print_costs, 
                                                                                        no_absval=no_absval, correlation_matrix=correlation_matrix, 
                                                                                        **kwargs)
                    merge = merge * len(self.graphs) 
                    self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
                    if qk_flag == True:
                        metric = self.metrics[f'qk{layer_no}']
                        correlation_matrix = self.cov_to_corr(metric['covariance'])
                        qk_merge, qk_unmerge, _, cost = get_merging_fn(mha_transform_fn)(n_heads, r=reduce_ratio, 
                                                                                permute_heads=permute_heads, head_assignments=attn_head_perm, 
                                                                                print_costs=print_costs, no_absval=no_absval, 
                                                                                correlation_matrix=correlation_matrix, **kwargs)
                        qk_merge = qk_merge * len(self.graphs)
                        self.merges[f'qk{layer_no}']  = qk_merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[f'qk{layer_no}'] = qk_unmerge.chunk(len(self.graphs), dim=0)
                else:
                    # if ignoring heads or non-mha merge matrix
                    merge, unmerge, _, cost = transform_fn(reduce_ratio, correlation_matrix=correlation_matrix, 
                                                    no_absval=no_absval, **kwargs)
                    merge = merge * len(self.graphs) 
                    self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)

                    if qk_flag:
                        metric = self.metrics[f'qk{layer_no}']
                        qk_merge, qk_unmerge, _, cost = transform_fn(reduce_ratio, print_costs=print_costs, no_absval=no_absval, 
                                                                correlation_matrix=correlation_matrix, **kwargs)
                        # add qk_merges to dict here so that attn merge can get added at end of block
                        qk_merge = qk_merge * len(self.graphs)
                        self.merges[f'qk{layer_no}']  = qk_merge.chunk(len(self.graphs), dim=1)
                        self.unmerges[f'qk{layer_no}'] = qk_unmerge.chunk(len(self.graphs), dim=0)
                
            # Handle FF
            else:
                # returns merge and unmerge matrixs
                merge, unmerge, _, cost = transform_fn(reduce_ratio, print_costs=print_costs, no_absval=no_absval, 
                                                    correlation_matrix=correlation_matrix,**kwargs)
                merge = merge * len(self.graphs) 
                self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
        
        elif contains_name(prev_node_layer, qk_nodes):
            continue
            # continuing because this is already handled in attention block

        # handle metrics associated with residuals here, other special cases
        else:
            info = self.graphs[0].get_node_info(node)
            print('res')
            print(info)
            if res == 'sep':
                correlation_matrix = corrs[node]
                merge, unmerge, _, cost = transform_fn(reduce_ratio, correlation_matrix=correlation_matrix, 
                                                no_absval=no_absval,**kwargs)
                merge = merge * len(self.graphs)
                self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)
            else:
                # res is first, last, or all:
                if global_res_merge == None:
                    correlation_matrix = corrs['res']
                    global_res_merge, global_res_unmerge, _, cost = transform_fn(reduce_ratio,  
                                                                        correlation_matrix=correlation_matrix, 
                                                                        no_absval=no_absval, **kwargs)
                    global_res_merge = global_res_merge * len(self.graphs)
                    self.merges[node] = global_res_merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = global_res_unmerge.chunk(len(self.graphs), dim=0)
                else: # merge was already learned
                    self.merges[node] = global_res_merge.chunk(len(self.graphs), dim=1)
                    self.unmerges[node] = global_res_unmerge.chunk(len(self.graphs), dim=0)
        cost_dict[node] = cost
    if qk_flag == True:
        for node in nodes:
            prev_node_layer = self.graphs[0].get_node_info(node-1)['layer']
            if prev_node_layer != None and contains_name(prev_node_layer, qk_nodes):
                layer_no =  [int(i) for i in self.graphs[0].get_node_info(node-1)['layer'].split('.') if i.isdigit()][0]
                self.merges[node] = self.merges[f'qk{layer_no}']
                self.unmerges[node] = self.unmerges[f'qk{layer_no}']
        for i in range(self.graphs[0].num_layers):
            self.merges.pop(f'qk{i}')
            self.unmerges.pop(f'qk{i}')
            
    self.compute_transform_time = time() - start_time
    return self.merges, self.unmerges, cost_dict