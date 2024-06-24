import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from mergekit.graph import Task
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from mergekit.metric_methods.all_metrics import Layer, Results

class MetricsHandler:
    """
    Object to handle metrics output. Allows for easy plotting of metrics by layer and across layers.

    Input:
        Use the load_metrics method to load the metrics into the handler.
        metrics: List of tasks and their metrics. This is the output of the run_measure function in mergekit.measure.

    Attributes:
        all_stats: Dictionary of recorded statistics for each layer. e.g. {'layer_name': {'cossim_mean': 0.5, 'cossim_std': 0.1}}
        stat_names: List of names of all statistics available. e.g. ['cossim_mean', 'cossim_std']
        layer_names: List of layer names. 

    Methods:
        load_metrics: Load the metrics into the handler.
        stats_at_layer: Get the metrics for a specific layer.
        info_at_layer: Get the weight info for a specific layer.
        line_plot: Plot a line plot of the chosen stat across layers.
        plot_node_hist: Plot a histogram of the stat for a specific layer.
    """
    def __init__(self):
        self.layer_names: List[str] = []
        self.results = Results()
        self.stat_names: List[str] = []

    def load_metrics(self, metrics: List[Tuple[Task, Layer]]):
        for task, metric in metrics:
            if metric is not None:
                self.results.add_layer(metric, name=task.weight_info.name)
                self.stat_names.extend(list(metric.metrics.keys()))
        self.layer_names = list(self.results.layers.keys())
        self.stat_names = list(set(self.stat_names))
    
    def stats_at_layer(self, layer_name: str) -> Dict[str, Any]:
        if layer_name not in self.results.layers:
            raise ValueError(f"Layer {layer_name} not found")
        return self.results.layers[layer_name]
    
    def info_at_layer(self, layer_name: str):
        if layer_name not in self.results.layers:
            raise ValueError(f"Layer {layer_name} not found")
        return self.results.layers[layer_name].weight_info
    
    def line_plot(self, stat: str, save_to:Optional[str]=None, **kwargs):
        fig, ax = plt.subplots()
        
        ax_kwargs = ['ylabel', 'title', 'ylim', 'xticklabels']
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in ax_kwargs}

        self._line_plot(ax, stat, plot_kwargs)
        self._set_plot_attributes(ax, stat, ax_kwargs, **kwargs)
        if save_to:
            plt.savefig(save_to)
        plt.show()
        plt.close()
    
    def categorise_layers(self, layer_names):
        # Hardcoded layernames for now - can be extended to include more categories or further generalised based on config
        categories = []
        for name in layer_names:
            if 'Attention Block' in name:
                categories.append('Attention Block')
            elif 'mlp' in name:
                categories.append('MLP')
            else:
                categories.append('Other')
        return categories

    def plotly_line_plot(self, stat: str, **kwargs):
        """
        Plot the stat values across layers using Plotly.

        Args:
            stat (str): The name of the stat to plot.
        
        Returns:
            List[go.Scatter]: List of Plotly Scatter objects.
        """

        if stat not in self.stat_names:
            print(f"Stat {stat} not found")
            return 

        means = [self.results.layers[layer].metrics[stat].mean_std.mean for layer in self.layer_names]
        stds  = [self.results.layers[layer].metrics[stat].mean_std.std for layer in self.layer_names]

        layer_categories = self.categorise_layers(self.layer_names)
        unique_categories = list(set(layer_categories))
        
        # Assign a unique color to each category
        cmap = plt.get_cmap('Set1', len(unique_categories))
        colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_categories))]

        category_styles = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

        traces = []

        for category in unique_categories:
            y_category = [means[i] if layer_categories[i] == category else None for i in range(len(self.layer_names))]
            std_category = [stds[i] if layer_categories[i] == category else None for i in range(len(self.layer_names))]
            
            traces.append(go.Scatter(
                x=self.layer_names,
                y=y_category,
                error_y=dict(
                    type='data',
                    array=std_category,
                    visible=True
                ),
                mode='markers',
                name=category,
                marker=dict(color=category_styles[category])
            ))
        return traces

    def _line_plot(self, ax, stat:str, plot_kwargs: Optional[Dict[str, Any]] = {}):
        """
        Plot the stat values with optional error bars.

        Args:
            ax: The matplotlib Axes object.
            stat (str): The name of the stat to plot.
            plot_kwargs: Additional keyword arguments for plotting.
        """
        means = [self.results.layers[layer].metrics[stat].mean for layer in self.layer_names]
        stds = [self.results.layers[layer].metrics[stat].std for layer in self.layer_names]
        ax.errorbar(self.layer_names, means, yerr=stds, fmt='-o', **plot_kwargs)

    def plot_node_heatmap(self, layer_name:str, stat:str):
        """
        Plot the stat values as a heatmap.
        
        Args:
            layer_name (str): The name of the layer.
            stat (str): The name of the stat to plot.
        Returns:
            go.Heatmap: Plotly Heatmap object.
        """
        heatmap = self.results.layers[layer_name].metrics[stat].heatmap.data

        return go.Heatmap(
            z=heatmap,
            colorscale='RdBu'
            )

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
    
    def plot_node_hist(self, layer_name: str, stat: str):

        hist = self.stats_at_layer(layer_name).metrics[stat].histogram
        count, edges, widths = hist.count, hist.edges, hist.widths
        return go.Bar(
            x=edges[:-1],
            y=count,
            width=widths,
            marker=dict(
                color='blue',
                line=dict(
                    color='black',
                    width=1
                )
            )
        )
    
    def node_plot_options(self, node):
        layer = self.results.layers[node]
            
        return [
            {"label": f"{metric.title()} Histogram", "value": [metric, 'histogram']}
            for metric in layer.metrics_with_property('histogram')
        ] + [
            {"label": f"{metric.title()} Heatmap", "value": [metric, 'heatmap']}
            for metric in layer.metrics_with_property('heatmap')
        ]


class ModelGraph:
    def __init__(self, metrics: List[Tuple['Task', Dict[str, Any]]]):
        self.metric_handler = MetricsHandler()
        self.metric_handler.load_metrics(metrics)
        self.hierarchy = []
        self.common_parts = self._find_common_parts()
        self.graph = nx.DiGraph()
        self._parse_task_names()

    def _find_common_parts(self) -> List[str]:
        """
        Find common parts in all task names.
        """
        common_parts = None
        for task_name in self.metric_handler.results.layers.keys():
            parts = task_name.split('.')
            if common_parts is None:
                common_parts = set(parts)
            else:
                common_parts.intersection_update(parts)

        return list(common_parts)
    def _parse_task_names(self):
        for task_name in self.metric_handler.results.layers.keys():
            self.hierarchy.append(task_name)

    def _add_nodes_and_edges(self, hierarchy):
        # Current implementation builds linear graph
        # Parallel paths (heads, skips) not yet supported
        prev = None
        for name in hierarchy:
            self.graph.add_node(name)
            if prev:
                self.graph.add_edge(prev, name)
            prev = name

    def construct_graph(self):
        self._add_nodes_and_edges(self.hierarchy)    

def create_app(nn_graph):
    app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'])

    app.layout = html.Div([
        create_header(),
        create_line_plot_section(nn_graph),
        create_node_metrics_section()
    ])

    register_callbacks(app, nn_graph)

    return app

def create_header():
    return html.H1('Network Weights Similarity Visualization', 
                   style={'textAlign': 'center', 'padding': '20px'})

def create_line_plot_section(nn_graph):
    return html.Div([
        dcc.Dropdown(
            id='line-plot-dropdown',
            options=[{'label': metric.replace('_', ' ').title(), 'value': metric} 
                    for metric in nn_graph.metric_handler.stat_names],
            value='cossim',
            style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'}
        ),
        dcc.Graph(id='line-plot', style={'width': '100%', 'height': '100vh'})
    ], className='container-fluid')

def create_node_metrics_section():
    return html.Div([
        html.H3('Node Metrics', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[],
            style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'},
            value=None
        ),
        dcc.Graph(id='node-details-plot', style={'width': '100%', 'height': '80vh', 'textAlign': 'center'})
    ], className='container-fluid')


def register_callbacks(app, nn_graph):
    @app.callback(
        Output('metric-dropdown', 'options'),
        Output('metric-dropdown', 'value'),
        Input('line-plot', 'clickData')
    )
    def update_metric_dropdown_options(clickData):
        if not clickData:
            return [], None

        try:
            node_name = clickData['points'][0]['x']
            options = nn_graph.metric_handler.node_plot_options(node_name)
            return options, options[0]['value'] if options else ([], None)
        
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error processing clickData: {e}")
            return [], None

    @app.callback(
        Output('node-details-plot', 'figure'),
        Input('metric-dropdown', 'value'),
        State('line-plot', 'clickData')
    )
    def display_node_data(selected_metric, clickData):
        if not clickData:
            return go.Figure()

        try:
            node_name = clickData['points'][0]['x']
            if not selected_metric:
                selected_metric = nn_graph.metric_handler.node_plot_options(node_name)[0]['value']

            metric_name, plot_type = selected_metric

            if 'histogram' in plot_type.lower():
                trace = nn_graph.metric_handler.plot_node_hist(node_name, stat=metric_name)
                return create_figure(trace, f"Histogram for {node_name} | {metric_name}", "Metric", "Value")
            elif 'heatmap' in plot_type.lower():
                trace = nn_graph.metric_handler.plot_node_heatmap(layer_name=node_name, stat=metric_name)
                return create_figure(trace, f"Heatmap for {node_name} | {metric_name}", "Model 1 Head", "Model 0 Head")
            
            return go.Figure()
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error processing node data: {e}")
            return go.Figure()

    @app.callback(
        Output('line-plot', 'figure'),
        Input('line-plot-dropdown', 'value')
    )
    def update_line_plot(selected_metric):
        if not selected_metric:
            return go.Figure()
        return go.Figure(data=nn_graph.metric_handler.plotly_line_plot(selected_metric))

def create_figure(trace, title, xaxis_title, yaxis_title):
    return go.Figure(data=[trace], layout=go.Layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    ))
