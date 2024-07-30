#%%
import torch
import h5py
import numpy as np
import click
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from mergekit.metric_methods.base import Results, Layer
from mergekit.metric_methods.aggregator_metrics import ModelAnalysisType, LayerComparisonType, METRICS_TABLE

from mergekit.architecture import WeightInfo

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
    
def valid_experiment(analysis_type, comparison_type):
    if comparison_type == LayerComparisonType.ALL_LAYERS:
        raise ValueError("Comparison type 'all_layers' is not supported")
    if analysis_type == ModelAnalysisType.COMPARISON and comparison_type in [LayerComparisonType.BLOCK, LayerComparisonType.SINGLE]:
        raise ValueError("Comparison type 'single' and 'block' only supported for individual analysis")
    if analysis_type == ModelAnalysisType.INDIVIDUAL and comparison_type == LayerComparisonType.CORRESPONDING_LAYERS:
        raise ValueError("Comparison type 'corresponding' only supported for comparison analysis")

# Experiment Loops
def single(representations_path, metric_classes, results, device='cpu'):
    if not results:
        results = Results()
    with LayerByIndex(representations_path) as representations:
        for layer in tqdm(representations, desc='Analysing Layer', 
                                     total=len(representations), leave=False, initial = 1):
            layer_name = layer.name.split('/')[-1]
            num = f"{int(layer_name.split('_')[-1]):03d}"
            layer_name = f"Layer {num}"
            metrics = [metric_class(device=device) for metric_class in metric_classes]

            for batch in tqdm(layer, desc='Batch', 
                                     total=len(layer), leave=False, initial = 1):
                batch = torch.tensor(layer[batch][:], device=device)
                
                # Calculate the metrics for each batch
                for metric in metrics:
                    metric.process_batch(batch)

            layer_results = Layer(WeightInfo(name=layer_name))
            # Aggregate over the batches and add to the layer results
            for metric in metrics:
                layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())
                # metric.clear()
            
            results.add_layer(layer_results, layer_name)

    return results

def corresponding(representations_path_0, representation_path_1, metric_classes, results, device='cpu'):
    if not results:
        results = Results()
    with LayerByIndex(representations_path_0) as representations_a, \
            LayerByIndex(representation_path_1) as representations_b:

        for layer_a, layer_b in tqdm(zip(representations_a, representations_b), 
                                    desc='Comparing Representations at layer', 
                                    total=len(representations_a), initial = 1):
            
            layer_a_name = layer_a.name.split('/')[-1]
            layer_b_name = layer_b.name.split('/')[-1]
            if layer_a_name != layer_b_name:
                raise ValueError(f'Layer mismatch: {layer_a_name} != {layer_b_name}')
            num = f"{int(layer_a_name.split('_')[-1]):03d}"
            layer_name = f"Layer {num}"

            metrics = [metric_class(device=device) for metric_class in metric_classes]

            for batch_a, batch_b in tqdm(zip(layer_a, layer_b), 
                                        desc='Batch', total=len(layer_a), leave=False, initial = 1):
                batch_a = torch.tensor(layer_a[batch_a][:], device=device)
                batch_b = torch.tensor(layer_b[batch_b][:], device=device)
                
                # Calculate the metrics for each batch
                for metric in metrics:
                    metric.process_batch(batch_a, batch_b)

            layer_results = Layer(WeightInfo(name=layer_name))
            # Aggregate over the batches and add to the layer results
            for metric in metrics:
                layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())
                # metric.clear()

            results.add_layer(layer_results, layer_name)
    return results

def block(representations_path, block_size, metric_classes, results, device='cpu'):
    if not results:
        results = Results()
    out = {metric().__class__.__name__.lower(): [] for metric in metric_classes}
    with LayerByIndex(representations_path) as reps:
        for idx, block_start in tqdm(enumerate(reps), desc=f'Comparing {block_size}-block, Block Start at Layer', 
                                     total=len(reps) - block_size, leave=False, initial = 1):
            if idx + block_size >= len(reps):
                break

            # Create metrics
            metrics = [metric_class(device=device) for metric_class in metric_classes]
            block_end = reps[idx + block_size]

            for batch_0, batch_1 in tqdm(zip(block_start, block_end), desc='Batch', 
                                         total=len(block_start), leave=False, initial = 1):
                batch_0 = torch.tensor(block_start[batch_0][:]).to(device)
                batch_1 = torch.tensor(block_end[batch_1][:]).to(device)
                
                for metric in metrics:
                    metric.process_batch(batch_0, batch_1)
            
            # Aggregate metrics and add to results
            layer_results = Layer(WeightInfo(name=f"Block {idx} size {block_size}"))
            for metric in metrics:
                # layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())
                out[metric.__class__.__name__.lower()].append(metric.aggregate())

            results.add_layer(layer_results, f"Block {idx} size {block_size}")
    for metric in out:
        out[metric] = np.array(out[metric])
    return out

def all_layers(representations_path_0, representations_path_1, metric_classes, results, device='cpu'):
    if not results:
        results = Results()
    with LayerByIndex(representations_path_0) as reps_0, LayerByIndex(representations_path_1) as reps_1:
        for idx_0, layer_0 in enumerate(tqdm(reps_0, desc='Model 0 Layers', 
                                     total=len(reps_0), leave=False, initial = 1)):
            for idx_1, layer_1 in enumerate(tqdm(reps_1, desc='Model 1 Layers', 
                                     total=len(reps_1), leave=False, initial = 1)):
                if len(layer_0) != len(layer_1):
                    raise ValueError(f'Layer mismatch: {len(layer_0)} != {len(layer_1)}')

                metrics = [metric_class(device=device) for metric_class in metric_classes]

                for batch_0, batch_1 in tqdm(zip(layer_0, layer_1), desc='Batch', 
                                         total=len(layer_0), leave=False, initial = 1):
                    batch_0 = torch.tensor(layer_0[batch_0][:]).to(device)
                    batch_1 = torch.tensor(layer_1[batch_1][:]).to(device)
                    
                    for metric in metrics:
                        metric.process_batch(batch_0, batch_1)
                
                layer_results = Layer(WeightInfo(name=f"Layer {idx_0} - Layer {idx_1}"))
                for metric in metrics:
                    layer_results.add_metric(metric.aggregate(), metric.__class__.__name__.lower())

                results.add_layer(layer_results, f"Layer {idx_0} - Layer {idx_1}")

    return results

@click.command()
@click.option('--config_yml', default="./representations/config.yml", help='path to the configuration file.')
def main(config_yml: str = "config.yml"):
    with open(config_yml, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    
    representation_paths = [Path(model_path) for model_path in config['representation_paths']]
    metrics_toggle = config['metrics']
    analysis_type = ModelAnalysisType(config['analysis_type'])
    comparison_type = LayerComparisonType(config['comparison_type'])

    out_dir = representation_paths[0].parent.parent / 'stored_results'



    device = 'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else \
             'cpu'


    use_metrics = {name: METRICS_TABLE[name] 
                   for name, enabled in metrics_toggle.items() 
                   if enabled}


    valid_experiment(analysis_type, comparison_type)

    final_results = []
    if analysis_type == ModelAnalysisType.INDIVIDUAL:
        out_paths = [out_dir / f"{str(rep).split('/')[-1].split('.')[0]}+{str(list(use_metrics.keys()))}.json" for rep in representation_paths]
        for out_path in out_paths:
            assert not Path(out_path).exists(), f'{out_path} already exists.'
        for representation_path in representation_paths:
            individual_results = Results()
            individual_results.model_paths = [representation_path]

            if not representation_path.exists():
                raise FileNotFoundError(f"Representation file {representation_path} not found")

            if comparison_type == LayerComparisonType.SINGLE:
                metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.SINGLE.value]]
                individual_results = single(representation_path, 
                                        metrics, 
                                        results=individual_results, 
                                        device=device)

            if comparison_type == LayerComparisonType.BLOCK:
                metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.BLOCK.value]]
                heatmaps = {}
                for metric in metrics:
                    heatmaps[metric().__class__.__name__.lower()] = np.array([])
                for block_size in range(1, 9):
                    block_res = block(representations_path=representation_path, 
                                        block_size=block_size, 
                                        metric_classes=metrics, 
                                        results=individual_results, 
                                        device=device)
                    for metric in metrics:
                        heatmaps[metric().__class__.__name__.lower()] = np.append(heatmaps[metric().__class__.__name__.lower()], block_res[metric().__class__.__name__.lower()])
                
                for metric in metrics:
                    result.across_layer_metrics[metric.__class__.__name__.lower()] = heatmaps[metric.__class__.__name__.lower()] # Definitely a simpler way to code this (X)

            if comparison_type == LayerComparisonType.CORRESPONDING_LAYERS:
                metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.CORRESPONDING_LAYERS.value]]
                individual_results = corresponding(representations_path_0=representation_path,
                                                    representation_path_1=representation_path,
                                                    metric_classes=metrics,
                                                    results=individual_results,
                                                    device=device)

            final_results.append(individual_results)
    
    if analysis_type == ModelAnalysisType.COMPARISON:
        out_paths = [out_dir / f"/{str([str(rep).split('/')[-1].split('.')[0] for rep in representation_paths])}+{str(list(use_metrics.keys()))}.json"]
        assert not Path(out_path).exists(), f'{out_path} already exists.'

        comparison_results = Results()
        comparison_results.model_paths = representation_paths

        if comparison_type == LayerComparisonType.CORRESPONDING_LAYERS:
            metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.CORRESPONDING_LAYERS.value]]
            comparison_results = corresponding(representations_path_0=representation_paths[0],
                                                representation_path_1=representation_paths[1],
                                                metric_classes=metrics,
                                                results=comparison_results,
                                                device=device)
        if comparison_type == LayerComparisonType.ALL_LAYERS:
            metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.ALL_LAYERS.value]]
            comparison_results = all_layers(representations_path_0=representation_paths[0],
                                    representations_path_1=representation_paths[1],
                                    metric_classes=metrics,
                                    results=comparison_results,
                                    device=device)
        final_results.append(comparison_results)

    for result, out_path in zip(final_results, out_paths):
        result.save(out_path) #SAVE AS WE GO, NOT ALL AT THE END (X)
            

if __name__ == "__main__":
    main()
