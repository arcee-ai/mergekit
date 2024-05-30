#%%
OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "./examples/linear_small.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

# actually do merge
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from mergekit.measure import run_measure
import matplotlib.pyplot as plt
import numpy as np
from mergekit.plot_tools.plot_tools import MetricsHandler, NeuralNetworkGraph
#%%

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

out = run_measure(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)
# %%

# %%
handler = MetricsHandler()
handler.load_metrics(out)
handler.stats()

# %%
handler.layers()
# %%
# handler.line_plot('cossim_mean', title='Cosine Similarity Mean', xticklabels=[f'layer {i}' for i in range(len(handler.layers()))],ylim=(0.90,1.0))#, xticklabels=[f'layer {i}' for i in range(len(handler.layers()))])
# %%



# Example usage:
metrics = handler.layers()
nn_graph = NeuralNetworkGraph([pair for pair in out if pair[1] is not None])
nn_graph.construct_graph()
nn_graph.plot_graph()
# %%
