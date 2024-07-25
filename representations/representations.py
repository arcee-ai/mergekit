#%%
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

import enum

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
    

class ModelAnalysisType(enum.Enum):
    INDIVIDUAL = "individual"
    COMPARISON = "comparison"


class LayerComparisonType(enum.Enum):
    SINGLE = "single" # Layer i
    BLOCK = "block" # Layer i in model 1 and layer i+(block size) in model 1
    CORRESPONDING_LAYERS = "corresponding_layers" # Layer i in model 1 and layer i in model 2
    ALL_LAYERS = "all_layers" # Layer i in model 1 and Layer j in model (1 or 2)


class MetricInput(enum.Enum):
    ONE_SHOT = "one_shot"  
    BATCHES = "batches"


def valid_experiment(analysis_type, comparison_type, metric_input):
    if comparison_type == LayerComparisonType.ALL_LAYERS:
        raise ValueError("Comparison type 'all_layers' is not supported")
    if analysis_type == ModelAnalysisType.COMPARISON and comparison_type in [LayerComparisonType.BLOCK, LayerComparisonType.SINGLE]:
        raise ValueError("Comparison type 'single' and 'block' only supported for individual analysis")
    if analysis_type == ModelAnalysisType.INDIVIDUAL and comparison_type == LayerComparisonType.CORRESPONDING_LAYERS:
        raise ValueError("Comparison type 'corresponding_layers' only supported for comparison analysis")


def layer_loader(representation_path):
    with LayerByIndex(representation_path) as representations:
        for layer in tqdm(representations, desc='Analysing Layer', 
                                     total=len(representations), leave=False, initial = 1):
            yield layer

def batch_loader(layer, device):
    for batch in tqdm(layer, desc='processing batch', 
                                     total=len(layer), leave=False, initial = 1):
        yield torch.tensor(layer[batch][:], device=device)

# Experiment Loops
def single(representation_path: str):
    for layer_idx, layer in enumerate(layer_loader(representation_path)):
        for batch in batch_loader(layer, device="cpu"):
            yield batch, layer_idx

def block(representation_path: str, block_size: int, device: str = "cpu"):
    with LayerByIndex(representation_path) as reps:
        for layer_idx, block_start in tqdm(enumerate(reps), desc=f'Comparing {block_size}-block, Block Start at Layer', 
                                     total=len(reps) - block_size, leave=False, initial = 1):
            if layer_idx + block_size >= len(reps):
                break

            block_end = reps[layer_idx + block_size]
            
            for batch_0, batch_1 in tqdm(zip(block_start, block_end), desc='Batch', 
                                         total=len(block_start), leave=False, initial = 1):
                batch_0 = torch.tensor(block_start[batch_0][:]).to(device)
                batch_1 = torch.tensor(block_end[batch_1][:]).to(device)
                yield (batch_0, batch_1), layer_idx

def corresponding_layers(representation_path_0: str, representation_path_1: str, device: str = "cpu"):
    with LayerByIndex(representation_path_0) as reps_0, LayerByIndex(representation_path_1) as reps_1:
        for layer_idx, (layer_0, layer_1) in enumerate(tqdm(zip(reps_0, reps_1), desc='Comparing Corresponding Layers', 
                                     total=len(reps_0), leave=False, initial = 1)):
            for batch_0, batch_1 in tqdm(zip(layer_0, layer_1), desc='Batch', 
                                         total=len(layer_0), leave=False, initial = 1):
                batch_0 = torch.tensor(layer_0[batch_0][:]).to(device)
                batch_1 = torch.tensor(layer_1[batch_1][:]).to(device)
                yield (batch_0, batch_1), layer_idx

def all_layers(representation_path_0: str, representation_path_1: str, device: str = "cpu"):
    with LayerByIndex(representation_path_0) as reps_0, LayerByIndex(representation_path_1) as reps_1:
        for layer_0_idx, layer_0 in enumerate(tqdm(reps_0, desc='Model 0 Layers', 
                                     total=len(reps_0), leave=False, initial = 1)):
            for layer_1_idx, layer_1 in enumerate(tqdm(reps_1, desc='Model 1 Layers', 
                                     total=len(reps_1), leave=False, initial = 1)):
                for batch_0, batch_1 in tqdm(zip(layer_0, layer_1), desc='Batch', 
                                         total=len(layer_0), leave=False, initial = 1):
                    batch_0 = torch.tensor(layer_0[batch_0][:]).to(device)
                    batch_1 = torch.tensor(layer_1[batch_1][:]).to(device)

                    yield (batch_0, batch_1), (layer_0_idx, layer_1_idx)
                
                
def main():
    representation_paths = [Path("/Users/elliotstein/Documents/Arcee/mergekit/representations/Representations_Qwen_Qwen2-7B-Instruct_microsoft_orca-math-word-problems-200k_4000.h5"),
                            Path("/Users/elliotstein/Documents/Arcee/mergekit/representations/Representations_arcee-ai_qwen2-7b-math-tess_microsoft_orca-math-word-problems-200k_4000.h5")
    ]

    analysis_type = ModelAnalysisType.INDIVIDUAL
    comparison_type = LayerComparisonType.SINGLE
    metric_input = MetricInput.BATCHES

    valid_experiment(analysis_type, comparison_type, metric_input)

    for data, layer_idx in single(representation_paths[0]):
        pass

    for data, layer_idx in block(representation_paths[0], 2):
        pass

    for data, layer_idx in corresponding_layers(representation_paths[0], representation_paths[1]):
        pass

    for data, layer_idx in all_layers(representation_paths[0], representation_paths[1]):
        pass
            
#%%

if __name__ == "__main__":
    main()

