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
from pathlib import Path
import pickle

class MetricMethod(MergeMethod):
    pass

# Structure of the results object


# Results
# ├── model_path: Optional[List[str]] # One for individual model, two for comparison
# └── layers: Dict[str, Layer]
#     └── Layer
#         ├── weight_info: WeightInfo (remove?)
#         └── metrics: Dict[str, Metric]
#             └── Metric
#                 ├── histogram: Optional[Histogram]
#                 ├── mean_std: Optional[MeanStd]
#                 ├── scatter_plot: Optional[ScatterPlot]
#                 └── heatmap: Optional[Heatmap]
from enum import Enum

class PlotType(Enum):
    HISTOGRAM = 'histogram'
    MEAN_STD = 'mean_std'
    SCATTER_PLOT = 'scatter_plot'
    HEATMAP = 'heatmap'

@dataclass
class MeanStd:
    mean: float
    std: Optional[float] = None

@dataclass
class Heatmap:
    data: torch.Tensor
    update_layout_options: Optional[Dict] = None

@dataclass
class Histogram:
    count: List[float]
    edges: List[float]
    widths: List[float]

@dataclass
class ScatterPlot:
    x: List[float]
    y: List[float]

@dataclass
class Metric:
    histogram: Optional[Histogram] = None
    mean_std: Optional[MeanStd] = None
    heatmap: Optional[Heatmap] = None
    scatter_plot: Optional[ScatterPlot] = None

    def filled_attributes(self) -> List[str]:
        filled_attrs = []
        for attr, value in self.__dict__.items():
            if value is not None:
                filled_attrs.append(attr)
        return filled_attrs

@dataclass
class Layer:
    weight_info: WeightInfo
    metrics: Dict[str, Metric] = field(default_factory=dict)

    def metrics_with_attribute(self, attribute: str) -> List[str]:
        return [name for name, metric in self.metrics.items() if attribute in metric.filled_attributes()]
    
    def add_metric(self, metric: Metric, name: str):
        if name not in self.metrics.keys():
            self.metrics[name] = metric
        else:
            raise ValueError(f"Metric with name {name} already exists in layer {self.weight_info.layer_name}.")

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

from typing import List, Tuple
from mergekit.graph import Task
    
class Results:
    # Class to store the statistics for each layer
    def __init__(self):
        self.layers: Dict[str, Layer] = {}
        self.across_layer_metrics: Dict[str, Metric] = {}
        self.model_paths: Optional[List[str]] = None
 
    def add_layer(self, layer: Layer, name: str):
        if name not in self.layers.keys():
            self.layers[name] = layer

    def load_metrics(self, metrics: List[Tuple[Task, Layer]], model_paths: Optional[List[str]] = None):
        self.model_paths = model_paths
        for task, metric in metrics:
            if metric is not None:
                self.add_layer(metric, name=task.weight_info.name)
        return self
    
    def get_lineplot_data(self, metric_name: str):
        means, stds = [],[]

        available_line_plots = self.available_plot_types(PlotType.MEAN_STD.value)
        if metric_name not in available_line_plots:
            return [], []

        layers_with_data = available_line_plots[metric_name]
        means = [self.layers[layer].metrics[metric_name].mean_std.mean for layer in layers_with_data]
        stds = [self.layers[layer].metrics[metric_name].mean_std.std for layer in layers_with_data]

        means = expand_to_fit(all_layer_names=list(self.layers.keys()), values=means, subset_layer_names=layers_with_data)
        stds = expand_to_fit(all_layer_names=list(self.layers.keys()), values=stds, subset_layer_names=layers_with_data)

        return means, stds
    
    def available_metrics(self) -> Dict[str, Dict[str, Any]]:
        all_metrics = set()
        for layer in self.layers.values():
            all_metrics.update(layer.metrics.keys())
        
        metric_info = {}
        for metric in all_metrics:
            info = {
                'layers': [],
                PlotType.MEAN_STD.value: False,
                PlotType.HISTOGRAM.value: False,
                PlotType.HEATMAP.value: False,
                PlotType.SCATTER_PLOT.value: False
            }
            for layer_name, layer in self.layers.items():
                if metric in layer.metrics:
                    info['layers'].append(layer_name)
                    m = layer.metrics[metric]
                    if m.mean_std:
                        info[PlotType.MEAN_STD.value] = True
                    if m.histogram:
                        info[PlotType.HISTOGRAM.value] = True
                    if m.heatmap:
                        info[PlotType.HEATMAP.value] = True
                    if m.scatter_plot:
                        info[PlotType.SCATTER_PLOT.value] = True
            metric_info[metric] = info
        return metric_info
    
    def available_plot_types(self, plot_type: str) -> Dict[str, List[str]]:
        # Returns dictionary with key metric_name and value: list of layers for which that metric has data
        metric_info = self.available_metrics()
        out = {}
        plot_type = 'mean_std' if plot_type == 'line_plot' else plot_type
        assert plot_type in [p.value for p in PlotType], f"Plot type {plot_type} is not valid. Must be one of {[p.value for p in PlotType]}"
        for metric_name, info in metric_info.items():
            if info[plot_type]:
                out[metric_name] = info['layers']
        return out

    def available_metrics_at_layer(self, layer_name: str) -> List[str]:
        if layer_name in self.layers:
            return list(self.layers[layer_name].metrics.keys())
        else:
            return []

    def print_metric_summary(self):
        metric_info = self.available_metrics()
        print("Available Metrics Summary:")
        for metric, info in metric_info.items():
            print(f"\nMetric: {metric}")
            print(f"  Has mean/std: {'Yes' if info[PlotType.MEAN_STD.value] else 'No'}")
            print(f"  Has histogram: {'Yes' if info[PlotType.HISTOGRAM.value] else 'No'}")
            print(f"  Has heatmap: {'Yes' if info[PlotType.HEATMAP.value] else 'No'}")
            print(f"  Has scatter plot: {'Yes' if info[PlotType.SCATTER_PLOT.value] else 'No'}")

    def finalise(self):
        self.layer_names = list(self.layers.keys())
        self.metric_names = list(set([metric for layer in self.layers.values() for metric in layer.metrics.keys()]))

    def save(self, path: str):
        path = Path(path)
        if not path.suffix or path.suffix != '.pkl':
            path = path.with_suffix('.pkl')
        
        with path.open('wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        path_obj = Path(path).resolve()
        if path_obj.exists() and path_obj.is_file():
            with open(path_obj, 'rb') as f:
                results = pickle.load(f)
            assert isinstance(results, Results), "Loaded object is not a Results object"
            return results
        else:
            raise FileNotFoundError(f"The path {path} does not exist or is not a file.")