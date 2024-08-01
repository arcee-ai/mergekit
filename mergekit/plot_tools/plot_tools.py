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
from mergekit.metric_methods.base import Results, PlotType
from mergekit.common import ModelReference
from plotly.subplots import make_subplots

global_colours_list = ['blue', 'red', 'green', 'purple', 'orange', 'pink']
global_shapes_list = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star']

class ResultsHandler:
    def __init__(self):
        self.intra_model_results: Dict[ModelReference, Results] = {}
        self.inter_model_results: Results = None
        self.available_layer_plots = {
            'mean_std': [],
            'histogram': [],
            'heatmap': [],
            'scatter_plot': []
        }

    def load_results(self, results: Results):
        results.finalise()
        if len(results.model_paths) == 2:
            self.inter_model_results = results
        elif len(results.model_paths) == 1:
            # key = results.model_paths[0]
            key = len(self.intra_model_results)
            self.intra_model_results[key] = results
        else:
            raise ValueError("Results should have either 1 or 2 model_paths")
        
        for plot_type in self.available_layer_plots.keys():

            if self.inter_model_results is not None:
                self.available_layer_plots[plot_type] += list(self.inter_model_results.available_plot_types(plot_type).keys())
            if self.intra_model_results is not None:
                for model_path, results in self.intra_model_results.items():
                    self.available_layer_plots[plot_type] += list(results.available_plot_types(plot_type).keys())
                
            self.available_layer_plots[plot_type] = list(set(self.available_layer_plots[plot_type]))
        
        self.all_results = list(self.intra_model_results.values()) + [self.inter_model_results]
        
    def categorise_layers(self, layer_names) -> List[str]:
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
        traces = []
        if metric_name in self.inter_model_results.available_plot_types('line_plot'): # bring if case into loop? (X)
            layer_names = self.inter_model_results.layer_names
            means, stds = self.inter_model_results.get_lineplot_data(metric_name)
            categorised_layers = self.categorise_layers(layer_names) # Different category for each layer type
            unique_categories = list(set(categorised_layers))
            traces = self._plotly_line_plot(layer_names, means, stds, categorised_layers, unique_categories) 

        else:
            unique_categories = list(self.intra_model_results.keys())
            for i, (model_path, results) in enumerate(self.intra_model_results.items()):
                layer_names = results.layer_names
                means, stds = results.get_lineplot_data(metric_name)
                if means:
                    categorised_layers = [model_path]*len(layer_names) # Different category for each model, every layer in each model has the same category
                    shape = global_shapes_list[i%len(global_shapes_list)]
                    traces.extend(self._plotly_line_plot(layer_names, means, stds, categorised_layers, unique_categories, shape))
                
        return traces, layer_names

    def _plotly_line_plot(self, x_values, means, stds, categorised_layers, unique_categories, shape:str='circle', **kwargs):
        """
        
        Returns:
            List[go.Scatter]: List of Plotly Scatter objects.
        """

        # Assign a unique color to each category
        cmap = plt.get_cmap('Set1', len(unique_categories))
        colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_categories))]

        category_styles = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

        traces = []

        for category in unique_categories:
            y_category = [means[i] if categorised_layers[i] == category else None for i in range(len(categorised_layers))]
            std_category = [stds[i] if categorised_layers[i] == category else None for i in range(len(categorised_layers))]
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
                name=str(category),
                marker=dict(color=category_styles[category]), 
                marker_symbol=shape
            ))
        return traces

    def plotly_layer_plot(self, layer_name:str, metric_name:str, plot_type:str):
        assert plot_type in [p.value for p in PlotType], f"Plot type {plot_type} not in {[p.value for p in PlotType]}"
        data = []

        for result in self.all_results:
            valid_metrics = result.available_plot_types(plot_type)
            if metric_name in valid_metrics.keys():
                if layer_name in valid_metrics[metric_name]:
                    data.append(getattr(result.layers[layer_name].metrics[metric_name], plot_type))
        
        return self.get_traces(data, plot_type) # Can prob use type of data to determine plot type (X)

    def get_traces(self, data:List, plot_type:str): # Can prob use type of data to determine plot type (X)
        if plot_type == PlotType.HEATMAP.value:
            traces = [go.Heatmap(
                z=d.data,
                colorscale='RdBu'
                ) for d in data]
        elif plot_type == PlotType.SCATTER_PLOT.value:
            traces = [go.Scatter(
                x = d.x,
                y = d.y,
                mode='markers'
            ) for d in data]
        elif plot_type == PlotType.HISTOGRAM.value:
            traces = []
            for i, d in enumerate(data):
                count, edges, widths = d.count, d.edges, d.widths
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
                        )))
        else: 
            raise ValueError(f'{plot_type} not valid for layer specific plot')

        return traces

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

    def layer_plot_options(self, layer_name: str):
        metric_options = []
        for plot_type in PlotType:
            if plot_type == PlotType.MEAN_STD:
                continue
            metric_options.extend([
                        {"label": f"{metric.title()} {plot_type.value}", "value": [metric, plot_type.value]}
                    for metric in self.inter_model_results.layers[layer_name].metrics_with_attribute(plot_type.value)]
                    )
            for result in self.all_results:
                if layer_name in result.layers:
                    metric_options.extend([
                        {"label": f"{metric.title()} {plot_type.value}", "value": [metric, plot_type.value]}
                    for metric in result.layers[layer_name].metrics_with_attribute(plot_type.value)]
                    )
                break # Assuming all intra-model results have the same metrics
        return metric_options


def create_app(results_handler):
    app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'])

    app.layout = html.Div([
        create_header(),
        create_line_plot_section(results_handler),
        create_single_layer_section(),
        create_across_layers_section(results_handler)
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
                    for metric_name in results_handler.available_layer_plots['mean_std']],
            value='cosine_similarity',
            style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'}
        ),
        dcc.Graph(id='line-plot', style={'width': '100%', 'height': '100vh'})
    ], className='container-fluid')

def create_single_layer_section():
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

def create_across_layers_section(results_handler):
    results = list(results_handler.intra_model_results.values()) + [results_handler.inter_model_results]

    plot_sections = []

    for result in results:
        if getattr(result, 'across_layer_metrics', None):
            for metric_name, metric in result.across_layer_metrics.items():
                for plot_type in ['histogram', 'heatmap', 'scatter_plot']:
                    if getattr(metric, plot_type, None): #(X) shouldn't need [0] - metric is being stored inside an array and shouldn't be!
                        plot_sections.append(html.Div([
                            html.H3(f'{plot_type+metric_name.replace("_", " ").title()} {plot_type.replace("_", " ").title()}', style={'textAlign': 'center'}),
                            dcc.Graph(id=f"{plot_type}-plot-{metric_name}-{str(result.model_paths[0].name).split('__')[-1].split('.')[0]}", style={'width': '50%', 'height': '50%', 'position': 'relative'})
                        ], className='container-fluid'))

    return html.Div(plot_sections)

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
    def update_metric_dropdown_options(clickData, selected_metric): # What distinguishes these options from layer-specific options?
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

            if plot_type.lower() in ["heatmap", "scatter_plot"]:
                xaxis_title = "Model 1"
                yaxis_title = "Model 0"
            elif plot_type.lower() == 'histogram':
                xaxis_title = "Value"
                yaxis_title = "Count"
            
            traces = results_handler.plotly_layer_plot(layer_name, metric_name, plot_type)
            
            return create_figure(traces=traces,
                                 title=f"{plot_type.title()} for {layer_name} | {metric_name}", 
                                 xaxis_title=xaxis_title,
                                 yaxis_title=yaxis_title,
                                 plot_type=plot_type
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
            if trace:
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

    for result in results_handler.all_results:
        if getattr(result, 'across_layer_metrics', None):
            for metric_name, metric in result.across_layer_metrics.items():
                for plot_type in ['histogram', 'heatmap', 'scatter_plot']:
                    if getattr(metric, plot_type, None): #(X) shouldn't need [0] - metric is being stored inside an array and shouldn't be!
                        id=f"{plot_type}-plot-{metric_name}-{str(result.model_paths[0].name).split('__')[-1].split('.')[0]}"

                        @app.callback(
                            Output(id, 'figure'),
                            Input(id, 'id')
                        )
                        def update_across_layers_plot(_id=id, plot_type=plot_type, metric=metric):
                            traces = results_handler.get_traces(data = [getattr(metric, plot_type)], plot_type = plot_type)
            
                            return create_figure(traces=traces,
                                                title=f"{id} | {metric_name}", 
                                                xaxis_title="Temp title",
                                                yaxis_title=metric_name,
                                                plot_type = plot_type
                                                )

def create_figure(traces, title, xaxis_title, yaxis_title, plot_type):
    if plot_type in ["scatter_plot", "heatmap"]:
        num_plots = len(traces)
        num_cols = 2
        num_rows = (num_plots + 1) // num_cols  

        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f"Plot {i+1}" for i in range(num_plots)])

        for i, trace in enumerate(traces):
            row = (i // num_cols) + 1
            col = (i % num_cols) + 1
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text=xaxis_title, row=row, col=col)
            fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
    else:
        fig = go.Figure()
        for trace in traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title=title,
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title)
        )
    
    return fig
