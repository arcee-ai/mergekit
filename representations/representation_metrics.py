#%%
import torch
import h5py
import numpy as np
import click
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from abc import ABC, abstractmethod

from mergekit.metric_methods.base import Results, Layer
from mergekit.metric_methods.aggregator_metrics import ModelAnalysisType, LayerComparisonType, METRICS_TABLE
from dataclasses import dataclass
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

def block(representations_path, block_size, metric_classes, device='cpu'):
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

@dataclass
class Configuration:
    representation_paths: List[Path]
    metrics: Dict[str, bool]
    analysis_type: ModelAnalysisType
    comparison_type: LayerComparisonType
    out_dir: Path
    device: str

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(
            representation_paths=[Path(path) for path in config_dict['representation_paths']],
            metrics=config_dict['metrics'],
            analysis_type=ModelAnalysisType(config_dict['analysis_type']),
            comparison_type=LayerComparisonType(config_dict['comparison_type']),
            out_dir=Path(config_dict['out_dir']),
            device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        ).validate()

    def validate(self):
        if self.comparison_type == LayerComparisonType.ALL_LAYERS:
            raise ValueError("Comparison type 'all_layers' is not supported")
        if self.analysis_type == ModelAnalysisType.COMPARISON and self.comparison_type in [LayerComparisonType.BLOCK, LayerComparisonType.SINGLE]:
            raise ValueError("Comparison type 'single' and 'block' only supported for individual analysis")
        if self.analysis_type == ModelAnalysisType.INDIVIDUAL and self.comparison_type == LayerComparisonType.CORRESPONDING_LAYERS:
            raise ValueError("Comparison type 'corresponding' only supported for comparison analysis")
        return self
        

class Experiment(ABC):
    @abstractmethod
    def run(self, config: Configuration):
        pass

class SingleExperiment(Experiment):
    def run(self, config: Configuration):
        # Implementation for single experiment
        use_metrics = {name: METRICS_TABLE[name] 
                   for name, enabled in config.metrics.items() 
                   if enabled}

        metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.SINGLE.value]]
        for representation_path in config.representation_paths:
            individual_results = Results()
            individual_results.model_paths = [representation_path]

            if not representation_path.exists():
                raise FileNotFoundError(f"Representation file {representation_path} not found")

            individual_results = single(representation_path, 
                                        metrics, 
                                        results=individual_results, 
                                        device=config.device)
            
            out_path = config.out_dir / f"{str(representation_path).split('/')[-1].split('.')[0]}+{str(list(use_metrics.keys()))}.json"
            individual_results.save(out_path)

class CorrespondingExperiment(Experiment):
    def run(self, config: Configuration):
        use_metrics = {name: METRICS_TABLE[name] 
                   for name, enabled in config.metrics.items() 
                   if enabled}

        metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.CORRESPONDING_LAYERS.value]]
        comparison_results.model_paths = [config.representation_paths[0], 
                                          config.representation_paths[0] if config.analysis_type == ModelAnalysisType.INDIVIDUAL.value else 
                                          config.representation_paths[1]]
        
        comparison_results = Results()
        for rep_0 in config.representation_paths:
            for rep_1 in config.representation_paths:
                if (rep_0 == rep_1 and config.analysis_type == ModelAnalysisType.INDIVIDUAL.value) or \
                    (rep_0 != rep_1 and config.analysis_type == ModelAnalysisType.COMPARISON.value):
                    
                    comparison_results = corresponding(representations_path_0=rep_0,
                                                        representation_path_1=rep_1,
                                                        metric_classes=metrics,
                                                        results=comparison_results,
                                                        device=config.device)
                    
                    out_path = config.out_dir / f"/{str([str(rep).split('/')[-1].split('.')[0] for rep in [rep_0, rep_1]])}+{str(list(use_metrics.keys()))}.json"
                    comparison_results.save(out_path)

class BlockExperiment(Experiment):
    def run(self, config: Configuration):
        use_metrics = {name: METRICS_TABLE[name] 
                   for name, enabled in config.metrics.items() 
                   if enabled}

        metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.BLOCK.value]]
        heatmaps = {}
        for representation_path in config.representation_paths:
            block_results = Results()
            block_results.model_paths = [representation_path]
            if not representation_path.exists():
                raise FileNotFoundError(f"Representation file {representation_path} not found")
            for metric in metrics:
                heatmaps[metric().__class__.__name__.lower()] = np.array([])
            for block_size in range(1, 9):
                block_res = block(representations_path=representation_path, 
                                    block_size=block_size, 
                                    metric_classes=metrics, 
                                    device=config.device)
                for metric in metrics:
                    heatmaps[metric().__class__.__name__.lower()] = np.append(heatmaps[metric().__class__.__name__.lower()], block_res[metric().__class__.__name__.lower()])
            
            for metric in metrics:
                block_results.across_layer_metrics[metric.__class__.__name__.lower()] = heatmaps[metric.__class__.__name__.lower()] # Definitely a simpler way to code this (X)

class AllLayersExperiment(Experiment):
    def run(self, config: Configuration):
        use_metrics = {name: METRICS_TABLE[name] 
                   for name, enabled in config.metrics.items() 
                   if enabled}

        metrics = [metric for metric in use_metrics.values() if metric().valid_for[LayerComparisonType.ALL_LAYERS.value]]
        comparison_results = Results()
        comparison_results.model_paths = config.representation_paths

        comparison_results = all_layers(representations_path_0=config.representation_paths[0],
                                representations_path_1=config.representation_paths[1],
                                metric_classes=metrics,
                                results=comparison_results,
                                device=config.device)
        
        out_path = config.out_dir / f"/{str([str(rep).split('/')[-1].split('.')[0] for rep in config.representation_paths])}+{str(list(use_metrics.keys()))}.json"
        comparison_results.save(out_path)
      

class ExperimentFactory:
    experiments: Dict[str, Experiment] = {
        "single": SingleExperiment,
        "corresponding": CorrespondingExperiment,
        "block": BlockExperiment,
        "all_layers": AllLayersExperiment,
    }

    @classmethod
    def create(cls, experiment_type: str) -> Experiment:
        experiment_class = cls.experiments.get(experiment_type)
        if not experiment_class:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        return experiment_class()

@click.command()
@click.option('--config_yml', default="./representations/config.yml", help='path to the configuration file.')
def main(config_yml: str = "config.yml"):
    config = yaml.safe_load(open(config_yml, 'r'))
    config['out_dir'] = Path(config['representation_paths'][0]).parent.parent / 'stored_results'
    config = Configuration.from_dict(config)

    experiment = ExperimentFactory.create(config.comparison_type.name.lower())
    experiment.run(config)

if __name__ == "__main__":
    main()
