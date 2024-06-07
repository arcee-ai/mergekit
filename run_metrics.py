#%%
OUTPUT_PATH = "./merged"  # folder to store the result in
CONFIG_YML = "./examples/linear_small.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.measure import run_measure
from mergekit.plot_tools.plot_tools import ModelGraph, create_app
#%%
with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    metric_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

if __name__ == '__main__':

    out = run_measure(
        metric_config,
        out_path=OUTPUT_PATH,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=COPY_TOKENIZER,
            lazy_unpickle=LAZY_UNPICKLE,
            low_cpu_memory=LOW_CPU_MEMORY,
        ),
    )

    nn_graph = ModelGraph([pair for pair in out if pair[1] is not None])
    nn_graph.construct_graph()

    app = create_app(nn_graph=nn_graph)
    app.run_server()