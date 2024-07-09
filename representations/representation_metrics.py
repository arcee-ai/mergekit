import torch
import h5py
import numpy as np
import click
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch.nn.functional as F

from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric, Results, Layer
from mergekit.metric_methods.metrics import cosine_similarity, smape, scale, mse, weight_magnitude, numerical_rank, compute_histogram, cosine_similarity_heatmap


from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.merge_methods.base import MergeMethod

class MetricAggregator:
    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        raise NotImplementedError

    def aggregate(self) -> Metric:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

class Cosine_Similarity(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cosine_similarities = torch.tensor([], device=self.device)

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_similarities = F.cosine_similarity(batch_a, batch_b, dim=1)
        self.cosine_similarities = torch.cat((self.cosine_similarities, batch_similarities))

    def aggregate(self) -> Metric:
        hist = compute_histogram(self.cosine_similarities, 100)        
        mean_std=MeanStd(
                mean=self.cosine_similarities.mean().item(), 
                std=self.cosine_similarities.std().item()
                )
        histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        self.clear()
        return Metric(
            histogram=histogram,
            mean_std=mean_std
        )

    def clear(self) -> None:
        self.cosine_similarities = torch.tensor([], device=self.device)

class MSE(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.square_errors = torch.tensor([], device=self.device)

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_square_errors = torch.square(batch_a - batch_b).flatten()
        self.square_errors = torch.cat((self.square_errors, batch_square_errors))

    def aggregate(self) -> Metric:
        hist = compute_histogram(self.square_errors, 100)
        mean_std=MeanStd(
                mean=self.square_errors.mean().item(),
                std=self.square_errors.std().item()
            )
        histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        self.clear()
        return Metric(
            histogram=histogram,
            mean_std=mean_std
        )

    def clear(self) -> None:
        self.square_errors = torch.tensor([], device=self.device)

class LayerByIndex:
    def __init__(self, reps_path: str):
        self.reps_path = reps_path
        self.representations = None
        self.layers = None

    def __enter__(self):
        self.representations = h5py.File(self.reps_path, 'r')
        self.layers = list(self.representations.keys())
        return self

    def __exit__(self, *args, **kwargs):
        if self.representations:
            self.representations.close()

    def __getitem__(self, idx: int):
        return self.representations[self.layers[idx]]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.representations[layer] for layer in self.layers)

def compare_representations(representations_a: h5py.File, representations_b: h5py.File, 
                            metrics_classes: Dict[str, MetricAggregator], device: str, results: Results) -> Dict[str, Any]:
    if results is None:
        results = Results()

    for layer_a, layer_b in tqdm(zip(representations_a, representations_b), 
                                 desc='Comparing Representations at layer', 
                                 total=len(representations_a), initial = 1):
        if layer_a != layer_b:
            raise ValueError(f'Layer mismatch: {layer_a} != {layer_b}')

        metrics = [metric_class(device=device) for metric_class in metrics_classes.values()]

        for batch_a, batch_b in tqdm(zip(representations_a[layer_a], representations_b[layer_b]), 
                                     desc='Batch', total=len(representations_a[layer_a]), leave=False, initial = 1):
            batch_a = torch.tensor(representations_a[layer_a][batch_a][:], device=device)
            batch_b = torch.tensor(representations_b[layer_b][batch_b][:], device=device)
            
            # Calculate the metrics for each batch
            for metric in metrics:
                metric.process_batch(batch_a, batch_b)

        layer_results = Layer(WeightInfo(name=layer_a))
        # Aggregate over the batches and add to the layer results
        for metric in metrics:
            layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())
            metric.clear()

        results.add_layer(layer_results, layer_a)

    return results

def compute_skip_block_metrics(reps_path: str, skip_layers: int, 
                               metric_classes: List[MetricAggregator], device: str) -> Results:
    results = Results()
    with LayerByIndex(reps_path) as reps:
        for idx, block_start in tqdm(enumerate(reps), desc=f'Comparing {skip_layers}-block, Block Start at Layer', 
                                     total=len(reps) - skip_layers, leave=False, initial = 1):
            if idx + skip_layers >= len(reps):
                break

            # Create metrics
            metrics = [metric_class(device=device) for metric_class in metric_classes]
            block_end = reps[idx + skip_layers]

            for batch_0, batch_1 in tqdm(zip(block_start, block_end), desc='Batch', 
                                         total=len(block_start), leave=False, initial = 1):
                batch_0 = torch.tensor(block_start[batch_0][:]).to(device)
                batch_1 = torch.tensor(block_end[batch_1][:]).to(device)
                
                for metric in metrics:
                    metric.process_batch(batch_0, batch_1)
            
            # Aggregate metrics and add to results
            layer_results = Layer(WeightInfo(name=f"Layer {idx}"))
            for metric in metrics:
                layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())

            results.add_layer(layer_results, f"Layer {idx}")

    return results

def results_list_to_heatmap(all_results, metric_names:List[str]) -> dict:
    rows = len(all_results)
    cols = max([len(result.layers) for result in all_results])
    heatmaps = {}
    for metric_name in metric_names:
        heatmap = np.full((rows, cols), np.nan)

        for i, result in enumerate(all_results):
            for j, layer in enumerate(result.layers):
                heatmap[i, j] = result.layers[layer].metrics[metric_name][0].mean_std.mean
        heatmaps[metric_name] = Heatmap(data=heatmap,
                                        update_layout_options = {
                                            'xaxis_title': 'Layer Number',
                                            'yaxis_title': 'Block Size',
                                        })
    return heatmaps

METRICS_TABLE = {
    'cosine_similarity': Cosine_Similarity,
    'mse': MSE
}

@click.command()
@click.option('--config_yml', default="./representations/config.yml", help='Merge configuration file.')
def main(config_yml: str):
    with open(config_yml, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    model_paths = config['representation_paths']
    metrics_toggle = config['metrics']
    skip_layers = config['block_analysis_parameters']['skip_layers']

    use_metrics = {name: METRICS_TABLE[name] for name, enabled in metrics_toggle.items() if enabled}

    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    
    for path in model_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    all_results = Results()
    if config['compare_between_models']:
        if len(model_paths) != 2:
            raise ValueError("Expected 2 model paths for comparison")

        with h5py.File(model_paths[0], 'r') as representations_a, \
             h5py.File(model_paths[1], 'r') as representations_b:

            all_results = compare_representations(representations_a, representations_b, 
                                              metrics_classes=use_metrics, device=device, results=all_results)

    if config['block_analysis']:
        for reps_path in tqdm(model_paths, desc='Model', leave=False, total=len(model_paths), initial = 1):
            results_list = []
            metric_classes = list(use_metrics.values())
            for skip_layer in tqdm(skip_layers, desc='Skip Layers', initial = 1):
                results_list.append(
                    compute_skip_block_metrics(reps_path, skip_layer, metric_classes=metric_classes, device=device)
                    )
            
            heatmaps = results_list_to_heatmap(results_list, metric_names=[metric.__name__.lower() for metric in metric_classes])
            for metric_name, heatmap in heatmaps.items():
                all_results.others[reps_path + '||' + metric_name] = heatmap

    all_results.save('results.pkl')

if __name__ == '__main__':
    main()