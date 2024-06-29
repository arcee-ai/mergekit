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
from mergekit.metric_methods.all_metrics import Layer
from mergekit.metric_methods.base import Results

class ResultsHandler:
    """
    Object to handle metrics results. Allows for easy plotting of metrics by layer and across layers.

    Input:
        Use the load_metrics method to load the metrics into the handler.
        metrics: List of tasks and their metrics. This is the output of the run_measure function in mergekit.measure.

    Attributes:
        all_stats: Dictionary of recorded statistics for each layer. e.g. {'layer_name': {'cossim_mean': 0.5, 'cossim_std': 0.1}}
        metric_names: List of names of all statistics available. e.g. ['cossim_mean', 'cossim_std']
        layer_names: List of layer names. 

    Methods:
        load_metrics: Load the metrics into the handler.
        # stats_at_layer: Get the metrics for a specific layer.
        # info_at_layer: Get the weight info for a specific layer.
        line_plot: Plot a line plot of the chosen stat across layers.
        plotly_layer_histogram: Plot a histogram of the stat for a specific layer.
    """
    def __init__(self, metrics: List[Tuple[Task, Layer]]):
        self.results = Results()
        self.load_metrics(metrics)

    def load_metrics(self, metrics: List[Tuple[Task, Layer]]):
        self.metric_names = []
        for task, metric in metrics:
            if metric is not None:
                self.results.add_layer(metric, name=task.weight_info.name)
                self.metric_names.extend(list(metric.metrics.keys()))
        self.layer_names = list(self.results.layers.keys())
        self.metric_names = list(set(self.metric_names))
        
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
            return []
        
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
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink'] # (X)

        traces = []
        for i, metric in enumerate(metric_list):
            hist = metric.histogram
            count, edges, widths = hist.count, hist.edges, hist.widths
            traces.append(go.Bar(
                x=edges[:-1],
                y=count,
                width=widths,
                marker=dict(
                    color=colors[i],
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

def create_app(results_handler):
    app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'])

    app.layout = html.Div([
        create_header(),
        create_line_plot_section(results_handler),
        create_layer_metrics_section()
    ])

    register_callbacks(app, results_handler)

    return app

def create_header():
    return html.H1('Network Weights Similarity Visualization', 
                   style={'textAlign': 'center', 'padding': '20px'})

def create_line_plot_section(results_handler):
    return html.Div([
        dcc.Dropdown(
            id='line-plot-dropdown',
            options=[{'label': metric_name.replace('_', ' ').title(), 'value': metric_name} 
                    for metric_name in results_handler.metric_names],
            value='cossim',
            style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'}
        ),
        dcc.Graph(id='line-plot', style={'width': '100%', 'height': '100vh'})
    ], className='container-fluid')

def create_layer_metrics_section():
    return html.Div([
        html.H3('Layer Metrics', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[],
            style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'},
            value=None
        ),
        dcc.Graph(id='layer-details-plot', style={'width': '100%', 'height': '80vh', 'textAlign': 'center'})
    ], className='container-fluid')

def default_option(options, current_value):
    if not options:
        return None
    if current_value.lower() in [o.lower() for o in options]:
        return current_value
    for option in options:
        if option.lower() in current_value.lower() or current_value.lower() in option.lower():
            return option
    return options[0]

def register_callbacks(app, results_handler):
    @app.callback(
        Output('metric-dropdown', 'options'),
        Output('metric-dropdown', 'value'),
        Input('line-plot', 'clickData'),
        Input('line-plot-dropdown', 'value')
    )
    def update_metric_dropdown_options(clickData, selected_metric):
        if not clickData:
            return [], None

        try:
            layer_name = clickData['points'][0]['x']
            options = results_handler.layer_plot_options(layer_name)
            default_label = default_option(list(map(lambda x: x['label'], options)), selected_metric)
            default_value = [option['value'] for option in options if option['label'] == default_label][0]
            return options, default_value
        
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error processing clickData: {e}")
            return [], None

    @app.callback(
        Output('layer-details-plot', 'figure'),
        Input('metric-dropdown', 'value'),
        State('line-plot', 'clickData')
    )
    def display_layer_data(selected_metric, clickData):
        if not clickData:
            return go.Figure()

        try:
            layer_name = clickData['points'][0]['x']
            if not selected_metric:
                selected_metric = results_handler.layer_plot_options(layer_name)[0]['value']

            metric_name, plot_type = selected_metric

            # Define default axis titles
            xaxis_title = "Value"
            yaxis_title = "Count"

            # Update axis titles if plot_type is 'heatmap'
            if plot_type.lower() == "heatmap":
                xaxis_title = "Model 1 Head"
                yaxis_title = "Model 0 Head"
            
            plot_function = {
                'histogram': results_handler.plotly_layer_histogram,
                'heatmap': results_handler.plotly_layer_heatmap
            }.get(plot_type.lower(), 
                  lambda *args, **kwargs: go.Figure()) # Defaults to *function* to produce empty figure
            
            traces = plot_function(layer_name=layer_name, 
                              metric_name=metric_name)
            
            return create_figure(traces=traces,
                                 title=f"{plot_type.title()} for {layer_name} | {metric_name}", 
                                 xaxis_title=xaxis_title,
                                 yaxis_title=yaxis_title
                                 )

        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error processing layer data: {e}")
            return go.Figure()

    @app.callback(
        Output('line-plot', 'figure'),
        Input('line-plot-dropdown', 'value')
    )
    def update_line_plot(selected_metric):
        if not selected_metric:
            return go.Figure()
        
        traces, layer_names = results_handler.plotly_line_plots(metric_name=selected_metric)
        fig = go.Figure()
        for trace in traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title=f"{selected_metric.replace('_', ' ').title()} Across Layers",
            xaxis=dict(
                title='Layer',
                tickvals=list(range(len(layer_names))), 
                ticktext=layer_names
            ),
            yaxis=dict(title=selected_metric.replace('_', ' ').title())
        )
        return fig

def create_figure(traces, title, xaxis_title, yaxis_title):
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title=yaxis_title)
    )
    
    return fig
