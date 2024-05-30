import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from mergekit.graph import Task
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class MetricsHandler():
    def __init__(self):
        self.all_metrics: Dict[str, Dict[str, Any]] = {}
        self.all_stats: List = []
        self.layer_names: List[str] = []

    def load_metrics(self, metrics: List[Tuple[Task, Dict[str, Any]]]):
        stats = set()
        for task, metric in metrics:
            if metric is not None:
                self.all_metrics[task.weight_info.name] = {'metric':metric,
                                                    'weight_info':task.weight_info}
                self.layer_names.append(task.weight_info.name)
                stats.update(metric.keys())
        
        self.all_stats = list(stats)

    def layers(self) -> List[str]:
        return self.layer_names
    
    def stats(self) -> List[str]:
        return self.all_stats
    
    def metric_at_layer(self, layer_name: str) -> Dict[str, Any]:
        if layer_name not in self.all_metrics:
            raise ValueError(f"Layer {layer_name} not found in metrics")
        return self.all_metrics[layer_name]['metric']
    
    def info_at_layer(self, layer_name: str):
        if layer_name not in self.all_metrics:
            raise ValueError(f"Layer {layer_name} not found in metrics")
        return self.all_metrics[layer_name]['weight_info']
    
    def line_plot(self, stat: str, save_to:Optional[str]=None, **kwargs):
        fig, ax = plt.subplots()
        
        ax_kwargs = ['ylabel', 'title', 'ylim', 'xticklabels']
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in ax_kwargs}

        self._plot_with_optional_error_bars(ax, stat, plot_kwargs)
        self._set_plot_attributes(ax, stat, ax_kwargs, **kwargs)
        if save_to:
            plt.savefig(save_to)
        plt.show()
        plt.close()

    def _plot_with_optional_error_bars(self, ax, stat:str, plot_kwargs: Optional[Dict[str, Any]] = {}):
        """
        Plot the stat values with optional error bars.

        Args:
            ax: The matplotlib Axes object.
            stat_values (List[float]): The values of the stat to plot.
            std_values (Optional[List[float]]): The standard deviation values for error bars.
            **kwargs: Additional keyword arguments for plotting.
        """
        std_values = None
        if f'{stat}_mean' in self.all_stats:
            std_stat = f"{stat}_std"
            stat = f'{stat}_mean'
            if std_stat in self.all_stats:
                std_values = [self.all_metrics[layer]['metric'].get(std_stat) for layer in self.layer_names]

        assert (stat in self.all_stats), f"Stat {stat} not found in metrics"
        stat_values = [self.all_metrics[layer]['metric'][stat] for layer in self.layer_names]
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
        ax.set_title(kwargs.get('title', f'{stat.capitalize()}'))

        # Set additional attributes
        for kwarg in ax_kwargs:
            if kwarg in kwargs:
                getattr(ax, f"set_{kwarg}")(kwargs[kwarg])


class NeuralNetworkGraph:
    def __init__(self, metrics: List[Tuple['Task', Dict[str, Any]]]):
        self.metrics = metrics
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
        if not self.metrics:
            return []

        common_parts = None
        for task, _ in self.metrics:
            parts = task.weight_info.name.split('.')
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
        for task, _ in self.metrics:
            name = task.weight_info.name
            self.hierarchy.append(name)

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
        pos = nx.planar_layout(self.graph)        
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
        
        metric_to_plot = [m for m in self.metric_handler.stats() if 'mean' in m][0]

        node_x = []
        node_y = []
        node_text = []
        node_values = []
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            metric_value = self.metric_handler.metric_at_layer(node)[metric_to_plot]
            node_text.append(f"{self._remove_common_parts(node)}: {metric_value:.2f}{'%' if 'SMAPE' in metric_to_plot else ''}")
            node_values.append(metric_value)

        # Normalize node values for coloring
        norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))
        node_colors = [norm(value) for value in node_values]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                cmin=min(node_values).item(),
                cmax=max(node_values).item(),
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Metric Value',
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
        fig.show()

