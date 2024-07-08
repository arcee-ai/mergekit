import torch
import h5py
import random
import numpy as np
import click
import yaml
import h5py

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.merge import run_merge
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler, global_colours_list

from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric, Results, Layer
from mergekit.metric_methods.metrics import cosine_similarity, smape, scale, mse, weight_magnitude, numerical_rank, compute_histogram, cosine_similarity_heatmap
from mergekit.architecture import WeightInfo


from typing import List, Tuple, Dict
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mergekit.graph import Task

class CustomResultsHandler(ResultsHandler):
    """
    Object to handle metrics results. Allows for easy plotting of metrics by layer and across layers.

    Input:
        Use the load_metrics method to load the metrics into the handler.
        metrics: List of tasks and their metrics. This is the output of the run_measure function in mergekit.measure.

    Attributes:
        all_stats: Dictionary of recorded statistics for each layer. e.g. {'layer_name': {'cosine_similarity_mean': 0.5, 'cosine_similarity_std': 0.1}}
        metric_names: List of names of all statistics available. e.g. ['cosine_similarity_mean', 'cosine_similarity_std']
        layer_names: List of layer names. 

    Methods:
        load_metrics: Load the metrics into the handler.
        # stats_at_layer: Get the metrics for a specific layer.
        # info_at_layer: Get the weight info for a specific layer.
        line_plot: Plot a line plot of the chosen stat across layers.
        plotly_layer_histogram: Plot a histogram of the stat for a specific layer.
    """
    def __init__(self):#, metrics: List[Tuple[Task, Layer]]):
        self.results = Results()
        # self.load_metrics(metrics)

    def load_metrics(self, metrics: List[Tuple[Task, Layer]]):
        self.metric_names = []
        for task, metric in metrics:
            if metric is not None:
                self.results.add_layer(metric, name=task.weight_info.name)
                self.metric_names.extend(list(metric.metrics.keys()))
        self.layer_names = list(self.results.layers.keys())
        self.metric_names = list(set(self.metric_names))

    def load_results(self, results: Results):
        self.results = results
        self.layer_names = list(self.results.layers.keys())
        self.metric_names = list(set([metric for layer in self.results.layers.values() for metric in layer.metrics.keys()]))
        
    def categorise_layers(self, layer_names):
        # Hardcoded layernames for now - can be extended to include more categories or further generalised based on config
        categories = []
        for name in layer_names:
            if 'Attention Block' in name:
                categories.append('Attention Block')
            elif 'mlp' in name:
                categories.append('MLP')
            elif 'layernorm' in name:
                categories.append('LayerNorm')
            else:
                categories.append('Other')
        return categories
    
    def plotly_line_plots(self, metric_name:str):
        if metric_name not in self.metric_names:
            print(f"Stat {metric_name} not found")
            return [], []
        
        layer_names = self.layer_names
        means, stds, model_refs = self.results.get_lineplot_data(metric_name)
        traces = []
        available_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star']

        if len(model_refs) > 1:
            unique_categories = [str(ref) for ref in model_refs]
            layer_categories = [[str(model_refs[i])]*len(layer_names) for i in range(len(model_refs))]
        else:
            layer_categories = [self.categorise_layers(layer_names)]
            unique_categories = list(set(layer_categories[0]))
        for i, model_ref in enumerate(model_refs):
            traces.extend(self._plotly_line_plot(layer_names, means[i], stds[i], layer_categories[i], unique_categories, shape=available_shapes[i%len(available_shapes)]))

        return traces, layer_names

    def _plotly_line_plot(self, x_values, means, stds, layer_categories, unique_categories, shape:str='circle', **kwargs):
        """
        Plot the stat values across layers using Plotly.

        Args:
            stat (str): The name of the stat to plot.
        
        Returns:
            List[go.Scatter]: List of Plotly Scatter objects.
        """

        # Assign a unique color to each category
        cmap = plt.get_cmap('Set1', len(unique_categories))
        colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_categories))]

        category_styles = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

        traces = []

        for category in unique_categories:
            y_category = [means[i] if layer_categories[i] == category else None for i in range(len(self.layer_names))]
            std_category = [stds[i] if layer_categories[i] == category else None for i in range(len(self.layer_names))]
            if all([y is None for y in y_category]):
                continue
            
            traces.append(go.Scatter(
                x=x_values,
                y=y_category,
                error_y=dict(
                    type='data',
                    array=std_category,
                    visible=True
                ),
                mode='markers',
                name=category,
                marker=dict(color=category_styles[category]), 
                marker_symbol=shape
            ))
        return traces

    def plotly_layer_heatmap(self, layer_name:str, metric_name:str):
        """
        Plot the stat values as a heatmap.
        
        Args:
            layer_name (str): The name of the layer.
            metric_name (str): The name of the stat to plot.
        Returns:
            go.Heatmap: Plotly Heatmap object.
        """
        metrics_list = self.results.layers[layer_name].metrics[metric_name]
        if len(metrics_list) > 1:
            raise Warning(f"Multiple heatmaps found for {metric_name} at layer {layer_name}. Using the first one.")
        
        heatmap = self.results.layers[layer_name].metrics[metric_name][0].heatmap.data

        return [go.Heatmap(
            z=heatmap,
            colorscale='RdBu'
            )]

    def _set_plot_attributes(self, ax, stat: str, ax_kwargs: List[str], **kwargs):
        """
        Set the attributes of the plot.

        Args:
            ax: The matplotlib Axes object.
            stat (str): The name of the stat.
            **kwargs: Additional keyword arguments for plot attributes.
        """
        # Defaults 
        ax.set_ylabel(kwargs.get('ylabel', stat))
        ax.set_xticks(np.arange(len(self.layer_names)))
        ax.set_xticklabels(self.layer_names, rotation=45)
        ax.set_title(kwargs.get('title', f'{stat.replace("_", " ").title()}'))

        # Set additional attributes
        for kwarg in ax_kwargs:
            if kwarg in kwargs:
                getattr(ax, f"set_{kwarg}")(kwargs[kwarg])
    
    def plotly_layer_histogram(self, layer_name: str, metric_name: str):
        metric_list = self.results.layers[layer_name].metrics[metric_name]

        traces = []
        for i, metric in enumerate(metric_list):
            hist = metric.histogram
            count, edges, widths = hist.count, hist.edges, hist.widths
            traces.append(go.Bar(
                x=edges[:-1],
                y=count,
                width=widths,
                marker=dict(
                    color=global_colours_list[i],
                    opacity=0.75,
                    line=dict(
                        color='black',
                        width=1
                    )
                ),
                name=str(metric.model_ref)
            ))
        return traces
    
    def layer_plot_options(self, layer_name: str):
        layer = self.results.layers[layer_name]
            
        return [
            {"label": f"{metric.title()} Histogram", "value": [metric, 'histogram']}
            for metric in layer.metrics_with_attribute('histogram')
        ] + [
            {"label": f"{metric.title()} Heatmap", "value": [metric, 'heatmap']}
            for metric in layer.metrics_with_attribute('heatmap')
        ]

@click.command()
@click.option('--results_path', 
              default="./representation_results_test.pkl", 
              help="path to load the results from.")
def main(results_path):
    results = Results()
    print('warning: results_path is hardcoded in main()')
    results_path = '/Users/elliotstein/Documents/Arcee/mergekit/representations/results_test.pkl'
    results = results.load(results_path)

    handler = CustomResultsHandler()
    handler.load_results(results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()