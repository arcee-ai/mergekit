import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from mergekit.graph import Task
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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
        self.all_stats: Dict[str, Dict[str, Any]] = {}
        self.stat_names: List = []
        self.layer_names: List[str] = []

    def load_metrics(self, metrics: List[Tuple[Task, Dict[str, Any]]]):
        for task, metric in metrics:
            if metric is not None:
                self.all_stats[task.weight_info.name] = {'metric':metric,
                                                    'weight_info':task.weight_info}
                self.layer_names.append(task.weight_info.name)
                self.stat_names.extend(metric.keys())
        
        self.stat_names = list(set(self.stat_names))
    
    def stats_at_layer(self, layer_name: str) -> Dict[str, Any]:
        if layer_name not in self.all_stats:
            raise ValueError(f"Layer {layer_name} not found")
        return self.all_stats[layer_name]['metric']
    
    def info_at_layer(self, layer_name: str):
        if layer_name not in self.all_stats:
            raise ValueError(f"Layer {layer_name} not found")
        return self.all_stats[layer_name]['weight_info']
    
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
    
    def plotly_line_plot(self, stat: str, save_to:Optional[str]=None, **kwargs):

        y = [self.all_stats[layer]['metric'][stat] for layer in self.layer_names]
        if f'{stat}'.replace('mean', 'std') in self.stat_names:
            std_stat = f'{stat}'.replace('mean', 'std')
            std_values = [self.all_stats[layer]['metric'].get(std_stat) for layer in self.layer_names] 

        return go.Scatter(
            x=self.layer_names,
            y=y,
            error_y=dict(
                type='data',
                array=std_values,
                visible=True
            ),
            mode='lines+markers',
            name='Line Plot'
        )

    def _line_plot(self, ax, stat:str, plot_kwargs: Optional[Dict[str, Any]] = {}):
        """
        Plot the stat values with optional error bars.

        Args:
            ax: The matplotlib Axes object.
            stat_values (List[float]): The values of the stat to plot.
            std_values (Optional[List[float]]): The standard deviation values for error bars.
            **kwargs: Additional keyword arguments for plotting.
        """
        std_values = None
        if f'{stat}_mean' in self.stat_names:
            std_stat = f"{stat}_std"
            stat = f'{stat}_mean'
            if std_stat in self.stat_names:
                std_values = [self.all_stats[layer]['metric'].get(std_stat) for layer in self.layer_names]

        assert (stat in self.stat_names), f"Stat {stat} not found"
        stat_values = [self.all_stats[layer]['metric'][stat] for layer in self.layer_names]
        if std_values:
            ax.errorbar(self.layer_names, stat_values, yerr=std_values, fmt='-o', **plot_kwargs)
        else:
            ax.plot(stat_values, **plot_kwargs)

    def heatmap_plot(self, layer_name:str, stat:str):
        """
        Plot the stat values as a heatmap.
        
        Args:
            layer_name (str): The name of the layer.
            stat (str): The name of the stat to plot.
        Returns:
            go.Heatmap: Plotly Heatmap object.
        """
        heatmap = self.all_stats[layer_name]['metric'][stat]

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

        bin_counts, bin_edges, bin_widths = self.stats_at_layer(layer_name)[stat].values()
        return go.Bar(
            x=bin_edges[:-1],
            y=bin_counts,
            width=bin_widths,
            marker=dict(
                color='blue',
                line=dict(
                    color='black',
                    width=1
                )
            )
        )


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
        for task_name, _ in self.metric_handler.all_stats.items():
            parts = task_name.split('.')
            if common_parts is None:
                common_parts = set(parts)
            else:
                common_parts.intersection_update(parts)

        return list(common_parts)

    def _remove_common_parts(self, name: str) -> str:
        """
        Remove common parts from the task name.
        """
        parts = name.split('.')
        cleaned_parts = [part for part in parts if part not in self.common_parts]
        return '.'.join(cleaned_parts)

    def _parse_task_names(self):
        for task_name, _ in self.metric_handler.all_stats.items():
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

    def plot_graph(self, colour_by='cossim_mean', save_to: str = None):
        """
        Plot the graph using Plotly for interactivity.
        """
        # Manually set positions for a straight line layout. 
        # Not yet implemented for more complex layouts with Parallel paths
        pos = {node: (i, i/5) for i, node in enumerate(self.graph.nodes())}

        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Find all metrics that contain 'mean' in their keys
        metrics_to_plot = [m for m in self.metric_handler.stat_names if 'mean' in m]

        node_x,node_y,node_text,hover_text = [], [], [], []
        node_values = {metric: [] for metric in metrics_to_plot}

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            metric_values = self.metric_handler.stats_at_layer(node)
            
            # Build the text for each node
            hover = self._remove_common_parts(node)
            for metric in metrics_to_plot:
                if metric in metric_values:
                    value = metric_values[metric]
                    hover += f"<br>{metric.replace('_', ' ').title()}: {value:.4f}{'%' if 'SMAPE' in metric else ''}"
                    node_values[metric].append(value)

            node_text.append(node)
            hover_text.append(hover)

        node_colors = [value for value in node_values[colour_by]]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                cmin=min(node_values[colour_by]),
                cmax=max(node_values[colour_by]),
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=colour_by.replace('_', ' ').title(),
                    xanchor='left',
                    titleside='right',
                ),
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)))

        if save_to:
            fig.write_html(save_to)
        return fig
    

def create_app(nn_graph):
    app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'])

    app.layout = html.Div([
        html.Div([
            html.H1('Network Weights Similarity Visualisation', style={'textAlign': 'center', 'padding': '20px'}),
            dcc.Dropdown(
                id='line-plot-dropdown',
                options=[{'label': metric.replace('_', ' ').title(), 'value': metric} for metric in nn_graph.metric_handler.stat_names if 'mean' in metric],
                value='cossim_mean',
                style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'}
            ),
            dcc.Graph(id='line-plot', style={'width': '100%', 'height': '100vh'}),
        ], className='container-fluid'),
        
        html.Div([
            html.H3('Node Metrics', style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[],
                style={'width': '50%', 'margin': 'auto', 'display': 'block', 'font-family': 'Arial'}
            ),
            dcc.Graph(id='node-details-plot', style={'width': '100%', 'height': '80vh', 'textAlign': 'center'}),
        ], className='container-fluid')
    ])

    @app.callback(
        Output('metric-dropdown', 'options'), Output('metric-dropdown', 'value'),
        [Input('line-plot', 'clickData')]
    )
    def update_metric_dropdown_options(clickData):
        if clickData is None:
            return [], None
        
        try:
            node_name = clickData['points'][0]['x']
            options = list(nn_graph.metric_handler.stats_at_layer(node_name).keys())
            options = [option for option in options if 'std' not in option]
            options = [
                {'label': option.replace('_', ' ').title(), 'value': option} for option in options if 'mean' not in option
            ]
            return options, options[0]['value'] if options else None
        
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing clickData: {e}")
            return [], None

    @app.callback(
        Output('node-details-plot', 'figure'),
        [Input('line-plot', 'clickData'), Input('metric-dropdown', 'value')],
    )
    def display_node_data(clickData, selected_metric):
        if clickData is None:
            return go.Figure()

        try:
            node_name = clickData['points'][0]['x']
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing clickData: {e}")
            return go.Figure()

        fig = go.Figure()
        if 'histogram' in selected_metric or 'Histogram' in selected_metric:
            trace = nn_graph.metric_handler.plot_node_hist(node_name, stat=selected_metric)
            fig.add_trace(trace)
            fig.update_layout(
                title=f"Metrics for {node_name} | {selected_metric}",
                xaxis_title="Metric",
                yaxis_title="Value"
            )
        elif 'heatmap' in selected_metric or 'Heatmap' in selected_metric:
            trace = nn_graph.metric_handler.heatmap_plot(layer_name=node_name, stat=selected_metric)
            fig.add_trace(trace)
            fig.update_layout(
                title=f"{node_name} | {selected_metric}",
                xaxis_title="Model 1 Head",
                yaxis_title="Model 0 Head"
            )

        return fig

    @app.callback(
        Output('line-plot', 'figure'),
        [Input('line-plot-dropdown', 'value')]
    )
    def update_line_plot(selected_metric):
        fig = go.Figure()
        fig.add_trace(nn_graph.metric_handler.plotly_line_plot(selected_metric))
        return fig
    return app
