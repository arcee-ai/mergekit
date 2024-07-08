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
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler, global_colours_list

from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric, Results, Layer
from mergekit.metric_methods.metrics import cosine_similarity, smape, scale, mse, weight_magnitude, numerical_rank, compute_histogram, cosine_similarity_heatmap
from mergekit.architecture import WeightInfo


from typing import List, Tuple, Dict
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mergekit.graph import Task

@click.command()
@click.option('--results_path', 
              default="./representation_results_test.pkl", 
              help="path to load the results from.")
def main(results_path):
    results = Results()
    print('warning: results_path is hardcoded in main()')
    results_path = '/Users/elliotstein/Documents/Arcee/mergekit/representations/results_test.pkl'
    results = results.load(results_path)

    handler = ResultsHandler()
    handler.load_results(results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()