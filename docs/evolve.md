# mergekit-evolve

`mergekit-evolve` is a script that uses an evolutionary algorithm (CMA-ES) to optimize the parameters of a merge against model metrics. This is inspired by SakanaAI's [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187), in particular their parameter-space approach. `mergekit-evolve` uses EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to define and evaluate the scoring function. The script is set up to be run either single-node or on a Ray cluster and has a few different strategies for scheduling operations depending on your particular configuration of compute.

## Installation

Install `mergekit` with the `evolve` (and optionally `vllm`) features:

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .[evolve,vllm]
```

If you had a perfectly good pytorch environment going and installing an older version of vLLM downgraded it and broke flash attention, run the following commands to fix it:

```sh
pip uninstall flash-attn
pip cache purge
pip install flash-attn
```

## Configuration

`mergekit-evolve` takes in a YAML configuration file that defines how the merge is parameterized and what metrics to optimize. The general syntax is as follows:

```yml
genome:
    models:
       - model_1
       - model_2
       ...
       - model_n
    merge_method: dare_ties
    base_model: base_model_if_needed
    tokenizer_source: null # optional
    layer_granularity: 8

    # optional:
    normalize: false
    allow_negative_weights: false
    smooth: false
    filters: ...
tasks:
  - name: lm_eval_task_name
    weight: 1.0 # optional
    metric: "acc,none" # defaults to acc,none
  - name: ... # as many as you want
```

### Genome Definition

The `genome` section of the configuration file defines the parameter space that `mergekit-evolve` will be optimizing in.

#### `models`

This should be a list of all of the models you want available to be merged. Depending on the merge method not all are guaranteed to be used in the final merge.

#### `merge_method`

Merge method to be used. Currently supported values are `linear`, `dare_ties`, `task_arithmetic`, `ties`, and `slerp`.

#### `base_model`

The base model for the merge, if applicable.

#### `layer_granularity`

A set of parameters will be introduced for each consecutive slice of `layer_granularity` layers. So for example, a 32-layer model like `mistralai/Mistral-7B-v0.1` with `layer_granularity: 8` will be divided into 4 groups of 8 layers with different merge parameters for each. The value specified here must be a divisor of the number of layers in your input models. Large values of `layer_granularity` will reduce the search space greatly, meaning you will get faster convergence at the cost of a potentially less good global solution.

When not set, one set of parameters will be used for all layers.

#### `normalize`

Sets the `normalize` flag when merging. For methods like `linear`, `ties`, and `dare_ties` this constrains the search space to a set of definitely valid models. Similarly to `layer_granularity`, this can greatly speed up convergence at the cost of ruling out oddball solutions that might score better than more standard merges.

#### `allow_negative_weights`

Pretty self explanatory. When this flag is not set, the absolute value of weight parameters is used. Sensible search space reduction for `linear` and `slerp`. For task arithmetic based methods you probably want `allow_negative_weights: true`.

#### `smooth`

If set to `true`, then parameter values will be interpolated across layers instead of assigning a single, fixed value to each block.

#### `filters`

Accepts a list of filters, as in `mergekit-yaml`, by which to separate the parameters. So, for example, setting filters as below for a Llama-based merge:

```yaml
filters:
  - self_attn
  - mlp
```

Will divide up the merge parameters into three groups - self attention parameters, MLP parameters, and a third for everything else. Separating the parameters out like this can be very beneficial when merging models trained on different prompt formats. It also makes your parameter space three times as big though!

### Task Definition

To evaluate the produced merges you need to specify a list of tasks supported by the EleutherAI LM evaluation harness. This can be either [built in tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) (don't be naughty) or tasks you define yourself (see the [New Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) for how). If your task does not use `acc` as the metric then you must specify the correct metric name. Each task can also optionally have a weight associated.

`mergekit-evolve` aims to maximize the score of the merge, so if you are using any tasks or metrics where a lower score is better (like perplexity) be sure to assign a negative weight to that task.

## Running `mergekit-evolve`

```sh
mergekit-evolve [OPTIONS] --storage-path PATH GENOME_CONFIG_PATH
```

`mergekit-evolve` needs a storage path specified, where it will save the input models, merges to evaluate, and the config for the current best merge evaluated. If you are not using in-memory merging this can require a _lot_ of space - expect at least one fp16 model per GPU.

Some important options:

### Scheduling Strategy (`--strategy`)

There are three different strategies implemented for scheduling merging and evaluation jobs.

#### `pool`

Assigns an actor to each GPU in your cluster and guarantees merges and evaluations are performed on the same node. This is a safe default suitable for any configuration, local or distributed.

#### `buffered`

Maintains a buffer of tasks scheduled to ensure that there is always a model mergign or ready to evaluate for each gpu. Allows for concurrent merging and evaluation of models on the same GPU if enough VRAM is available. Only suitable for a single-node setup or when `--storage-path` points to a fast shared filesystem.

#### `serial`

Uses Ray placement groups to ensure merges and their evaluations happen on the same node, but otherwise just lets Ray take the wheel. Maybe give a try if you're having trouble with the other two, otherwise probably don't use it.

### Evaluation LLM Backend

By default `mergekit-evolve` will use the `hf` backend for `lm-eval`. To use vLLM instead, pass the `--vllm` flag.

### On-Disk vs. In-Memory

By default `mergekit-evolve` will perform merges, write the result to disk, then start up an instance of lm-eval pointing at that path. This is a safe default and will generally always work but also causes a lot of GPU downtime and eats disk space. When using the `pool` scheduling strategy, you have the option to instead keep a model resident in memory and directly update its parameters instead of merging to disk. This is much faster and uses no additional disk space. However, it does involve mucking around in the internals of vLLM and the LM evaluation harness. So it might break at any moment! Choose wisely. Use `--in-memory` to enable this mode.

### Task search path

If you're using custom task definitions (and you should be) then you can append to the search path using the `--task-search-path` option. This should point to the directory your custom task YAML is in (or a parent of that directory). Multiple paths can be included by repeating the option.

### Batch size

Override the batch size used during merge evaluation. If using vLLM `auto` is recommended (default).

### CMA-ES options

#### `--max-fevals`

Maximum number of merges to evaluate. Note that the `cma` package is very loosey-goosey with this number and will happily go over by 50% depending on the size of each generation. Set to 100 by default.

#### `--sigma0`

Initial value of sigma for CMA-ES. No need to play with this unless you really know what you're doing.

### WandB logging

`mergekit-evolve` supports logging metrics to Weights & Biases. Enable this functionality with the `--wandb` flag. Project and entity names can be overridden with the `--wandb-project` and `--wandb-entity` options.

### Example

```sh
mergekit-evolve --strategy pool --wandb --wandb-project mergekit-evolve --wandb-entity arcee-ai --storage-path /path/to/mergekit-evolve/ ./config.yml
```

## Output

`mergekit-evolve` will write the merge configuration for the best merge found so far to the storage path with the filename `best_config.yaml`. If you're using WandB it will also log the config as an artifact. The script will keep running until a KeyboardInterrupt is received or `--max-fevals` is generously exceeded.

## Caveats

`mergekit-evolve` is a work in progress and has probably not been tested on your specific configuration. Keep an eye on the output before leaving it running, and if you run in to any issues don't hesitate to file an issue!

## Acknowledgements

Thanks to SakanaAI for the inspiration and the EleutherAI team for the LM evaluation harness.
