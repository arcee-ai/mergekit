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
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler

from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric, Results, Layer
from mergekit.metric_methods.metrics import cosine_similarity, smape, scale, mse, weight_magnitude, numerical_rank, compute_histogram, cosine_similarity_heatmap
from mergekit.architecture import WeightInfo
from typing import List

from tqdm import tqdm

from pathlib import Path

import torch.nn.functional as F

class MetricAggregator():
    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor):
        pass

    def aggregate(self):
        pass

    def clear(self):
        self.__init__()

class Cosine_Similarity(MetricAggregator):
    def __init__(self, device="cpu"):
        self.cosine_similarities = torch.tensor([]).to(device)

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor):
        batch_similarities = F.cosine_similarity(batch_a, batch_b, dim=1)
        self.cosine_similarities = torch.cat((self.cosine_similarities, batch_similarities), dim=0)

    def aggregate(self):
        hist = compute_histogram(self.cosine_similarities, 100)        
        return Metric(
            mean_std=MeanStd(
                mean=self.cosine_similarities.mean().item(), 
                std=self.cosine_similarities.std().item()),
            histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        )

class MSE(MetricAggregator):
    def __init__(self, device="cpu"):
        self.square_errors = torch.tensor([]).to(device)


    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor):
        assert batch_a.size(1) == batch_b.size(1)
        batch_square_errors = torch.square(batch_a - batch_b).flatten()
        self.square_errors = torch.cat((self.square_errors, batch_square_errors), dim=0)

    def aggregate(self):
        hist = compute_histogram(self.square_errors, 100)
        out = Metric(
            mean_std=MeanStd(
                mean=self.square_errors.mean().item(),
                std=self.square_errors.std().item()
            ),
            histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        )
        self.clear()
        return out

class LayerByIndex:
    def __init__(self, reps_path):
        self.reps_path = reps_path
        self.representations = None
        self.layers = None
        self.iter_index = 0
    
    def __enter__(self):
        self.representations = h5py.File(self.reps_path, 'r')
        self.layers = list(self.representations.keys())
        return self
    
    def __exit__(self, *args, **kwargs):
        if self.representations:
            self.representations.close()
    
    def __getitem__(self, idx):
        return self.representations[self.layers[idx]]
    
    def __len__(self):
        return len(self.layers)
    
    def __iter__(self):
        self.iter_index = 0
        return self
    
    def __next__(self):
        if self.iter_index < len(self.layers):
            layer = self.representations[self.layers[self.iter_index]]
            self.iter_index += 1
            return layer
        else:
            raise StopIteration
    
    def batches_in_layer(self, idx):
        return len(self.representations[self.layers[idx]])

def compare_representations(representations_a: h5py.File, representations_b: h5py.File, metrics_classes, device: str):
    results=Results()
    
    # Compare corresponding layers from both models
    for layer_a, layer_b in tqdm(zip(representations_a, representations_b), desc='Comparing Represenations at layer', total=len(representations_a), leave=False):
        metrics = [metric_class(device=device) for metric_class in metrics_classes.values()]

        layer_results = Layer(WeightInfo(name=layer_a))
        if layer_a != layer_b:
            raise ValueError(f'Layer mismatch: {layer_a} != {layer_b}')
        
        # Load the representations
        layer_representations_a = representations_a[layer_a]
        layer_representations_b = representations_b[layer_b]

        for batch_a, batch_b in tqdm(zip(layer_representations_a, layer_representations_b), desc='Batch', total=len(layer_representations_a), leave=False):
            batch_a = torch.tensor(layer_representations_a[batch_a][:], device=device)
            batch_b = torch.tensor(layer_representations_b[batch_b][:], device=device)
            # Calculate the metrics for each batch
            for metric in metrics:
                metric.process_batch(batch_a, batch_b)
        
        # Aggregate over the batches and add to the layer results
        for metric in metrics:
            layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower().lower())
            metric.clear()


        # Add the layer to the results
        results.add_layer(layer_results, layer_a)
    
    return results

def compute_skip_block_metrics(reps_path:str, skip_layers:int, metric_classes:List[MetricAggregator], device:str='cpu'):
    results = Results()
    with LayerByIndex(reps_path) as reps:
        for idx, block_start in tqdm(enumerate(reps), desc=f'Comparing {skip_layers}-block, Block Start at Layer', total=len(reps)-skip_layers, leave=False):
            # Create metrics
            metrics = [metric_class(device=device) for metric_class in metric_classes]

            if idx + skip_layers >= len(reps):
                continue
            block_end = reps[idx + skip_layers]

            # Each metric processes every batch
            for batch_0, batch_1 in tqdm(zip(block_start, block_end), desc='Batch', total=len(block_start), leave=False):
                batch_0 = torch.tensor(block_start[batch_0][:]).to(device)
                batch_1 = torch.tensor(block_end[batch_1][:]).to(device)

                for metric in metrics:
                    metric.process_batch(batch_0, batch_1)
            
            # Aggregate metrics and add to results
            layer_results = Layer(WeightInfo(name=f"{idx}"))
            for metric in metrics:
                layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower().lower())
            results.add_layer(layer_results, f"Layer {idx}")
            
            # Clear memory from metrics
            for metric in metrics:
                metric.clear()
    return results

def resultslist_to_heatmap(all_results, metric_names:List[str]) -> dict:
    rows = len(all_results)
    cols = max([len(result.layers) for result in all_results])
    heatmaps = {}
    for metric_name in metric_names:
        heatmap = np.full((rows, cols), np.nan)


        for i, result in enumerate(all_results):
            # row = len(all_results) - (i+1)
            for j, layer in enumerate(result.layers):
                heatmap[i, j] = result.layers[layer].metrics[metric_name][0].mean_std.mean
        heatmaps[metric_name] = Heatmap(data=heatmap)
    return heatmaps

metrics_table = {
    'cosine_similarity': Cosine_Similarity,
    'mse': MSE
}

@click.command()
@click.option('--config_yml', 
              default="./config.yml",
              help='merge configuration file.')
def main(config_yml):
    with open(config_yml, "r", encoding="utf-8") as fp:
        config_yml = yaml.safe_load(fp)

    model_paths = config_yml['representation_paths']
    metrics_toggle = config_yml['metrics']
    skip_layers = config_yml['block_analysis_parameters']['skip_layers']

    results = Results()

    use_metrics = {}
    for metric in metrics_table:
        if metrics_toggle[metric]:
            use_metrics[metric] = metrics_table[metric]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    for path in model_paths:
        assert Path(path).exists(), f"File not found: {path}"

    if config_yml['compare_between_models']:
        if len(model_paths) != 2:
            raise ValueError("Expected 2 model paths for comparison")

        with h5py.File(model_paths[0], 'r') as representations_a, \
            h5py.File(model_paths[1], 'r') as representations_b:

            # Compare corresponding layer representations 
            results = compare_representations(representations_a, 
                                    representations_b, 
                                    metrics_classes=use_metrics,
                                    device=device)
            
            results.save('results_compare')

    if config_yml['block_analysis']:
        # Analyse individual layer representations
        for reps_path in model_paths:
            
            metric_classes = list(use_metrics.values())

            all_results = []
            for skip_layer in tqdm(skip_layers, desc='Skip Layers', total=len(skip_layers)):
                all_results.append(
                    compute_skip_block_metrics(reps_path, skip_layer, metric_classes, device=device)
                )
            
            heatmaps = resultslist_to_heatmap(all_results, metric_names=[metric.__name__.lower() for metric in metric_classes])

            for metric_name, heatmap in heatmaps.items():
                results.others[reps_path + '||' + metric_name] = heatmap

        results.save('results_block_analysis')

    results.save('results')

if __name__ == '__main__':
    main()