#%%
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
from mergekit.metric_methods.metrics import cossim, smape, scale, mse, weight_magnitude, numerical_rank, compute_histogram, cossim_heatmap
from mergekit.architecture import WeightInfo

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

class CosineSimilarity(MetricAggregator):
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

        # CHECK DIMENSIONALITY (X)

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

@click.command()
@click.option('--reps_a_path', 
              default="NEW_Representations_BEE-spoke-data_smol_llama-220M-GQA_train_4000.h5", 
              help="path to load first set of representations from.")
@click.option('--reps_b_path',
                default="NEW_Representations_BEE-spoke-data_smol_llama-220M-openhermes_train_4000.h5",
                help="path to load second set of representations from.")
def main(reps_a_path, reps_b_path):
    results = Results()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    reps_a_path = Path(__file__).parent / reps_a_path
    reps_b_path = Path(__file__).parent / reps_b_path

    assert reps_a_path.exists(), f"File not found: {reps_a_path}"
    assert reps_b_path.exists(), f"File not found: {reps_b_path}"
    
    with h5py.File(reps_a_path, 'r') as representations_a, \
         h5py.File(reps_b_path, 'r') as representations_b:

        for layer_a, layer_b in tqdm(zip(representations_a, representations_b), desc='Layers', total=len(representations_a)):
            metrics = {
                'Cosine Similarity' : CosineSimilarity(device=device),
                'MSE' : MSE(device=device)
            }

            layer_results = Layer(WeightInfo(name=layer_a))
            if layer_a != layer_b:
                raise ValueError(f'Layer mismatch: {layer_a} != {layer_b}')
            
            # Load the representations
            layer_representations_a = representations_a[layer_a]
            layer_representations_b = representations_b[layer_b]

            for batch_a, batch_b in tqdm(zip(layer_representations_a, layer_representations_b), desc='Batches', total=len(layer_representations_a), leave=False):
                batch_a = torch.tensor(layer_representations_a[batch_a][:], device=device)
                batch_b = torch.tensor(layer_representations_b[batch_b][:], device=device)
                # Calculate the metrics for each batch
                for _, metric in metrics.items():
                    metric.process_batch(batch_a, batch_b)
            
            # Aggregate over the batches and add to the layer results
            for name, metric in metrics.items():
                layer_results.add_metric(metric.aggregate(), name)
                metric.clear()


            # Add the layer to the results
            results.add_layer(layer_results, layer_a)

    results.save('results_test')

if __name__ == '__main__':
    main()