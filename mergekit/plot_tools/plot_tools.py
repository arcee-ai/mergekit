import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from mergekit.graph import Task
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class MetricsHandler():
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

        self._plot(ax, stat, plot_kwargs)
        self._set_plot_attributes(ax, stat, ax_kwargs, **kwargs)
        if save_to:
            plt.savefig(save_to)
        plt.show()
        plt.close()

    def _plot(self, ax, stat:str, plot_kwargs: Optional[Dict[str, Any]] = {}):
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
        ax.set_title(kwargs.get('title', f'{stat.replace("_", " ").capitalize()}'))

        # Set additional attributes
        for kwarg in ax_kwargs:
            if kwarg in kwargs:
                getattr(ax, f"set_{kwarg}")(kwargs[kwarg])
    
    def plot_node_hist(self, layer_name: str, stat: str):

        bin_counts, bin_edges, bin_widths = self.all_stats[layer_name]['metric'][stat].values()
        # Create a bar chart using Plotly
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

    def plot_graph(self, save_to: str = None):
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
                    hover += f"<br>{metric.replace('_', ' ').capitalize()}: {value:.4f}{'%' if 'SMAPE' in metric else ''}"
                    node_values[metric].append(value)

            node_text.append(node)
            hover_text.append(hover)

        first_metric = metrics_to_plot[0]
        node_colors = [value.item() for value in node_values[first_metric]]

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
                cmin=min(node_values[first_metric]).item(),
                cmax=max(node_values[first_metric]).item(),
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=first_metric.replace('_', ' ').capitalize(),
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
    

