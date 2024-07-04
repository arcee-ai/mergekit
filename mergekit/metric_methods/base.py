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

from typing import Any, List, Optional, Dict

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference

from mergekit.merge_methods.base import MergeMethod
from dataclasses import dataclass, field
from collections import defaultdict
import torch

import pickle

class MetricMethod(MergeMethod):
    pass

# Structure of the results object

# Results
# └── layers: Dict[str, Layer]
#     └── Layer
#         ├── weight_info: WeightInfo
#         └── metrics: Dict[str, List[Metric]]
#             └── Metric
#                 ├── histogram: Optional[Histogram]
#                 ├── mean_std: Optional[MeanStd]
#                 ├── heatmap: Optional[Heatmap]
#                 └── model_ref: Optional[ModelReference]

# Each Layer stores metrics under a key (e.g. 'cosine_similarity') in a dictionary.
# The values stored under each key are a **list** of Metric objects. This is to allow for a single metric type to be computed for each model.
    # For metrics which compare between models, (e.g. cosine similarity) the list will contain a single Metric object storing the comparison data.
    # For metrics which analyse individual models, (e.g. intrinsic dimension) the list will contain a Metric object for each model.


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
    histogram: Optional[Histogram] = None
    mean_std: Optional[MeanStd] = None
    heatmap: Optional[Heatmap] = None
    model_ref: Optional[ModelReference] = None # For intra-model metrics.

    def filled_attributes(self) -> List[str]:
        filled_attrs = []
        for attr, value in self.__dict__.items():
            if value is not None:
                filled_attrs.append(attr)
        return filled_attrs

@dataclass
class Layer:
    weight_info: WeightInfo
    metrics: Dict[str, List[Metric]] = field(default_factory=dict)

    def metrics_with_attribute(self, attribute: str) -> List[str]:
        return [name for name, metric in self.metrics.items() if attribute in metric[0].filled_attributes()]
    
    def add_metric(self, metric: Metric, name: str):
        if name not in self.metrics.keys():
            self.metrics[name] = [metric]
        else:
            self.metrics[name].append(metric)

    def add_metric_list(self, metric_list: List[Metric], name: str):
        for metric in metric_list:
            self.add_metric(metric, name)

def expand_to_fit(all_layer_names: List[str], values: List[float], subset_layer_names: List[str]) -> List[float]:
    """
    Expands a list of values to fit a larger list of layer names, filling in missing values with None.

    Args:
        all_layer_names (List[str]): List of all layer names.
        values (List[float]): List of values to expand.
        subset_layer_names (List[str]): List of layer names that the values correspond to.

    Returns:
        List[float]: Expanded list of values, with None values for missing layers.
    """
    result = [None] * len(all_layer_names)
    subset_dict = dict(zip(subset_layer_names, values))
    
    for i, layer in enumerate(all_layer_names):
        if layer in subset_dict:
            result[i] = subset_dict[layer]
    
    return result
    
class Results:
    # Class to store the statistics for each layer
    def __init__(self):
        self.layers: Dict[str, Layer] = {}

    def add_layer(self, layer: Layer, name: str):
        if name not in self.layers.keys():
            self.layers[name] = layer

    def get_metric(self, layer_name: str, metric_name: str) -> Metric:
        return self.get_layer(layer_name, metric_name)
    
    def get_lineplot_data(self, metric_name: str):
        means, stds = defaultdict(list), defaultdict(list)
        layers = []

        for name, layer in self.layers.items():
            if metric_name in layer.metrics:    
                for model_result in layer.metrics[metric_name]:
                    model_ref = model_result.model_ref if model_result.model_ref else 'all'
                    means[model_ref].append(model_result.mean_std.mean)
                    stds[model_ref].append(model_result.mean_std.std)
                layers.append(name)

        means_list, stds_list, model_references = list(means.values()), list(stds.values()), list(means.keys())
        for i, model_ref in enumerate(model_references):
            means_list[i] = expand_to_fit(all_layer_names=list(self.layers.keys()), values=means_list[i], subset_layer_names=layers)
            stds_list[i] = expand_to_fit(all_layer_names=list(self.layers.keys()), values=stds_list[i], subset_layer_names=layers)

        return means_list, stds_list, model_references
    
    def available_metrics(self) -> Dict[str, Dict[str, Any]]:
        all_metrics = set()
        for layer in self.layers.values():
            all_metrics.update(layer.metrics.keys())
        
        metric_info = {}
        for metric in all_metrics:
            info = {
                'layers': [],
                'has_mean_std': False,
                'has_histogram': False,
                'has_heatmap': False,
                'has_model_ref': False
            }
            for layer_name, layer in self.layers.items():
                if metric in layer.metrics:
                    info['layers'].append(layer_name)
                    for m in layer.metrics[metric]:
                        if m.mean_std:
                            info['has_mean_std'] = True
                        if m.histogram:
                            info['has_histogram'] = True
                        if m.heatmap:
                            info['has_heatmap'] = True
                        if m.model_ref:
                            info['has_model_ref'] = True
            metric_info[metric] = info
        return metric_info

    def print_metric_summary(self):
        metric_info = self.available_metrics()
        print("Available Metrics Summary:")
        for metric, info in metric_info.items():
            print(f"\nMetric: {metric}")
            # print(f"  Available in layers: {', '.join(info['layers'])}")
            print(f"  Has mean/std: {'Yes' if info['has_mean_std'] else 'No'}")
            print(f"  Has histogram: {'Yes' if info['has_histogram'] else 'No'}")
            print(f"  Has heatmap: {'Yes' if info['has_heatmap'] else 'No'}")
            print(f"  Has model reference: {'Yes' if info['has_model_ref'] else 'No'}")

    def save(self, path: str):
        path = path + '.pkl' if not path.endswith('.pkl') else path
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)