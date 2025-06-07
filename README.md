# mergekit

`mergekit` is a toolkit for merging pre-trained language models. `mergekit` uses an out-of-core approach to perform unreasonably elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, with more coming as they catch my attention.

## Contents

- [Why Merge Models?](#why-merge-models)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Usage](#usage)
- [Merge Configuration](#merge-configuration)
  - [Parameter Specification](#parameter-specification)
  - [Tokenizer Configuration](#tokenizer-configuration)
  - [Chat Template Configuration](#chat-template-configuration)
  - [Examples](#examples)
- [Merge Methods](#merge-methods)
- [LoRA Extraction](#lora-extraction)
- [Mixture of Experts Merging](#mixture-of-experts-merging)
- [Evolutionary Merge Methods](#evolutionary-merge-methods)
- [Multi-Stage Merging (`mergekit-multi`)](#multi-stage-merging-mergekit-multi)
- [Raw PyTorch Model Merging (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch)
- [Tokenizer Transplantation (`mergekit-tokensurgeon`)](#tokenizer-transplantation-mergekit-tokensurgeon)
- [Merge in the Cloud](#-merge-in-the-cloud-)
- [Citation](#citation)

## Why Merge Models?

Model merging is a powerful technique that allows combining the strengths of different models without the computational overhead of ensembling or the need for additional training. By operating directly in the weight space of models, merging can:

- Combine multiple specialized models into a single versatile model
- Transfer capabilities between models without access to training data
- Find optimal trade-offs between different model behaviors
- Improve performance while maintaining inference costs
- Create new capabilities through creative model combinations

Unlike traditional ensembling which requires running multiple models, merged models maintain the same inference cost as a single model while often achieving comparable or superior performance.

## Features

Key features of `mergekit` include:

- Supports Llama, Mistral, GPT-NeoX, StableLM, and more
- Many [merge methods](#merge-methods)
- GPU or CPU execution
- Lazy loading of tensors for low memory use
- Interpolated gradients for parameter values (inspired by Gryphe's [BlockMerge_Gradient](https://github.com/Gryphe/BlockMerge_Gradient) script)
- Piecewise assembly of language models from layers ("Frankenmerging")
- [Mixture of Experts merging](#mixture-of-experts-merging)
- [LORA extraction](#lora-extraction)
- [Evolutionary merge methods](#evolutionary-merge-methods)
- [Multi-stage merging](#multi-stage-merging-mergekit-multi) for complex workflows.
- [Merging of raw PyTorch models (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch).

🌐 GUI Launch Alert 🤗 - We are excited to announce the launch of a mega-GPU backed graphical user interface for mergekit in Arcee! This GUI simplifies the merging process, making it more accessible to a broader audience. Check it out and contribute at the [Arcee App](https://app.arcee.ai). There is also a [Hugging Face Space](https://huggingface.co/mergekit-community) with limited amounts of GPUs.

## Installation

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

If the above fails with the error of:

```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`

## Contributing

We welcome contributions to `mergekit`! If you have ideas for new merge methods, features, or other improvements, please check out our [contributing guide](CONTRIBUTING.md) for details on how to get started.

## Usage

The script `mergekit-yaml` is the main entry point for `mergekit`. It takes a YAML configuration file and an output path, like so:

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [--lazy-unpickle] [--allow-crimes] [... other options]
```

This will run the merge and write your merged model to `./output-model-directory`.

For more information on the arguments accepted by `mergekit-yaml` run the command `mergekit-yaml --help`.

### Uploading to Huggingface

When you have a merged model you're happy with, you may want to share it on the Hugging Face Hub. `mergekit` generates a `README.md` for your merge with some basic information for a model card. You can edit it to include more details about your merge, like giving it a good name or explaining what it's good at; rewrite it entirely; or use the generated `README.md` as-is. It is also possible to edit your `README.md` online once it has been uploaded to the Hub.

Once you're happy with your model card and merged model, you can upload it to the Hugging Face Hub using the [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index) Python library.

```sh
# log in to huggingface with an access token (must have write permission)
huggingface-cli login
# upload your model
huggingface-cli upload your_hf_username/my-cool-model ./output-model-directory .
```

The [documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-upload) for `huggingface_hub` goes into more detail about other options for uploading.

## Merge Configuration

Merge configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer` or `tokenizer_source`: Determines how to construct a tokenizer for the merged model.
- `chat_template`: Specifies a chat template for the merged model.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `models.*.parameters` or `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall

### Tokenizer Configuration

The tokenizer behavior can be configured in two ways: using the new `tokenizer` field (recommended) or the legacy `tokenizer_source` field (maintained for backward compatibility). These fields are mutually exclusive - you should use one or the other, not both.

#### Modern Configuration (tokenizer)

The `tokenizer` field provides fine-grained control over vocabulary and embeddings:

```yaml
tokenizer:
  source: "union"  # or "base" or a specific model path
  tokens:          # Optional: configure specific tokens
    <token_name>:
      source: ...  # Specify embedding source
      force: false # Optional: force this embedding for all models
  pad_to_multiple_of: null  # Optional: pad vocabulary size
```

##### Tokenizer Source

The `source` field determines the vocabulary of the output model:

- `union`: Combine vocabularies from all input models (default)
- `base`: Use vocabulary from the base model
- `"path/to/model"`: Use vocabulary from a specific model

##### Token Embedding Handling

When merging models with different vocabularies, mergekit uses smart defaults to handle token embeddings:

- If a token exists in the base model, its embedding is used as the default
- If only one model has the token, that model's embedding is used
- Otherwise, an average of all available embeddings is used

You can override these defaults for specific tokens:

```yaml
tokenizer:
  source: union
  tokens:
    # Use embedding from a specific model
    <|im_start|>:
      source: "path/to/chatml/model"

    # Force a specific embedding for all models
    <|special|>:
      source: "path/to/model"
      force: true

    # Map a token to another model's token embedding
    <|renamed_token|>:
      source:
        kind: "model_token"
        model: "path/to/model"
        token: "<|original_token|>"  # or use token_id: 1234
```

##### Practical Example

Here's how you might preserve both Llama 3 Instruct and ChatML prompt formats when merging models:

```yaml
tokenizer:
  source: union
  tokens:
    # ChatML tokens
    <|im_start|>:
      source: "chatml_model"
    <|im_end|>:
      source: "chatml_model"

    # Llama 3 tokens - force original embeddings
    <|start_header_id|>:
      source: "llama3_model"
      force: true
    <|end_header_id|>:
      source: "llama3_model"
      force: true
    <|eot_id|>:
      source: "llama3_model"
      force: true
```

#### Legacy Configuration (tokenizer_source)

For backward compatibility, the `tokenizer_source` field is still supported:

```yaml
tokenizer_source: "union"  # or "base" or a model path
```

This provides basic tokenizer selection but lacks the fine-grained control of the modern `tokenizer` field.

### Chat Template Configuration

The optional `chat_template` field allows overriding the chat template used for the merged model.

```yaml
chat_template: "auto"  # or a template name or Jinja2 template
```

Options include:

- `"auto"`: Automatically select the most common template among input models
- Built-in templates: `"alpaca"`, `"chatml"`, `"llama3"`, `"mistral"`, `"exaone"`
- A Jinja2 template string for custom formatting

### Examples

Several examples of merge configurations are available in [`examples/`](examples/).

## Merge Methods

`mergekit` offers many methods for merging models, each with its own strengths and weaknesses. Choosing the right method depends on your specific goals, the relationship between the models you're merging, and the desired characteristics of the final model.

For detailed explanations, parameter descriptions, and use cases for each method, please see our [**Merge Method Guide**](docs/merge_methods.md).

### Method Overview

| Method (`value`)                                                                                                      | Core Idea                                                            | # Models | Base Model | Key Strengths / Use Cases                                       |
|:----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:--------:|:----:|:---------------------------------------------------------------|
| [**Linear** (`linear`)](docs/merge_methods.md#linear-linear)                                                          | Simple weighted average of model parameters.                         |    ≥2    |  -   | Averaging similar checkpoints, model soups.                     |
| [**SLERP** (`slerp`)](docs/merge_methods.md#slerp-slerp)                                                              | Spherical linear interpolation between two models.                   |     2    |  ✓   | Smoothly transitioning between two models.                      |
| [**NuSLERP** (`nuslerp`)](docs/merge_methods.md#nuslerp-nuslerp)                                                        | Enhanced SLERP with flexible weighting.                              |     2    |  *   | More intuitive SLERP; task vector SLERP.                        |
| [**Multi-SLERP** (`multislerp`)](docs/merge_methods.md#multi-slerp-multislerp)                                          | Barycentric SLERP for multiple models.                               |    ≥2    |  *   | Spherical interpolation for >2 models.                          |
| [**Karcher Mean** (`karcher`)](docs/merge_methods.md#karcher-mean-karcher)                                              | Riemannian barycenter of model parameters.                           |    ≥2    |  -   | Geometrically sound averaging on manifolds.                     |
| [**Task Arithmetic** (`task_arithmetic`)](docs/merge_methods.md#task-arithmetic-task_arithmetic)                      | Linearly combine "task vectors" (differences from a base).           |    ≥2    |  ✓   | Transferring/combining fine-tuned skills.                       |
| [**TIES** (`ties`)](docs/merge_methods.md#ties-merging-ties)                                                          | Task arithmetic + sparsification & sign consensus.                   |    ≥2    |  ✓   | Merging many models, reducing interference.                     |
| [**DARE** (`dare_linear`, `dare_ties`)](docs/merge_methods.md#dare-dare_linear-dare_ties)                               | Task arithmetic + random pruning & rescaling.                        |    ≥2    |  ✓   | Robust skill retention, similar to TIES.                        |
| [**DELLA** (`della`, `della_linear`)](docs/merge_methods.md#della-della-della_linear)                                   | Task arithmetic + adaptive magnitude-based pruning.                  |    ≥2    |  ✓   | Prioritizing important changes, reducing interference.          |
| [**Model Breadcrumbs** (`breadcrumbs`, `breadcrumbs_ties`)](docs/merge_methods.md#model-breadcrumbs-breadcrumbs_ties)   | Task arithmetic + outlier removal (small & large diffs).             |    ≥2    |  ✓   | Refining task vectors by removing extreme changes.              |
| [**SCE** (`sce`)](docs/merge_methods.md#sce-sce)                                                                      | Task arithmetic + adaptive matrix-level weighting based on variance. |    ≥2    |  ✓   | Dynamically weighting models based on parameter variance.       |
| [**Model Stock** (`model_stock`)](docs/merge_methods.md#model-stock-model_stock)                                        | Geometric weight calculation for linear interpolation.               |    ≥3    |  ✓   | Finding good linear interpolation weights for many checkpoints. |
| [**Nearswap** (`nearswap`)](docs/merge_methods.md#nearswap-nearswap)                                                    | Interpolate where parameters are similar.                            |     2    |  ✓   | Selective merging based on parameter similarity.                |
| [**Arcee Fusion** (`arcee_fusion`)](docs/merge_methods.md#arcee-fusion-arcee_fusion)                                    | Dynamic thresholding for fusing important changes.                   |     2    |  ✓   | Identifying and merging salient features.                       |
| [**Passthrough** (`passthrough`)](docs/merge_methods.md#passthrough-passthrough)                                        | Directly copies tensors from a single input model.                      |     1    |  -   | Frankenmerging, layer stacking, model surgery.                  |

**Key for `Base Model` Column:**

- ✓: **Required** - One of the input models *must* be designated as the `base_model`.
- *: **Optional** - One of the input models *can* be designated as the `base_model`.
- -: **Not Applicable** - `base_model` has no effect on this method.

## LoRA Extraction

Mergekit allows extracting PEFT-compatible low-rank approximations of finetuned models.

### Usage

```sh
mergekit-extract-lora --model finetuned_model_id_or_path --base-model base_model_id_or_path --out-path output_path [--no-lazy-unpickle] [--cuda] [--max-rank=desired_rank] [--sv-epsilon=tol]
```

## Mixture of Experts Merging

The `mergekit-moe` script supports merging multiple dense models into a mixture of experts, either for direct use or for further training. For more details see the [`mergekit-moe` documentation](docs/moe.md).

## Evolutionary Merge Methods

See [`docs/evolve.md`](docs/evolve.md) for details.

## Multi-Stage Merging (`mergekit-multi`)

`mergekit-multi` enables the execution of complex, multi-stage model merging workflows. You can define multiple merge configurations in a single YAML file, where later merges can use the outputs of earlier ones as inputs. This is useful for building up sophisticated models through a series of targeted merges.

See the [`mergekit-multi` documentation](docs/multimerge.md) for usage details and examples.

## Raw PyTorch Model Merging (`mergekit-pytorch`)

For merging arbitrary PyTorch models (not necessarily Hugging Face Transformers), `mergekit-pytorch` provides a way to apply mergekit's algorithms directly to `.pt` or `.safetensors` checkpoints. The configuration is similar to the YAML format used in `mergekit-yaml`, but does not support layer slicing or tokenizer configuration.

### Usage

```sh
mergekit-pytorch path/to/your/raw_config.yml ./output_pytorch_model_directory [options]
```

Use `mergekit-pytorch --help` for detailed options.

## Tokenizer Transplantation (`mergekit-tokensurgeon`)

`mergekit-tokensurgeon` is a specialized tool for transplanting tokenizers between models, allowing you to align the vocabulary of one model with another. This is particularly useful for cheaply producing draft models for speculative decoding or for cross-tokenizer knowledge distillation. See the [documentation](docs/tokensurgeon.md) for more details and how to use it.

## ✨ Merge in the Cloud ✨

We host merging on Arcee's cloud GPUs - you can launch a cloud merge in the [Arcee App](https://app.arcee.ai). Or through python - grab an ARCEE_API_KEY:

`export ARCEE_API_KEY=<your-api-key>`
`pip install -q arcee-py`

```python
import arcee
arcee.merge_yaml("bio-merge","./examples/bio-merge.yml")
```

Check your merge status at the [Arcee App](https://app.arcee.ai)

When complete, either deploy your merge:

```python
arcee.start_deployment("bio-merge", merging="bio-merge")
```

Or download your merge:

`!arcee merging download bio-merge`

## Citation

If you find `mergekit` useful in your research, please consider citing the [paper](https://aclanthology.org/2024.emnlp-industry.36/):

```bibtex
@inproceedings{goddard-etal-2024-arcees,
    title = "Arcee{'}s {M}erge{K}it: A Toolkit for Merging Large Language Models",
    author = "Goddard, Charles  and
      Siriwardhana, Shamane  and
      Ehghaghi, Malikeh  and
      Meyers, Luke  and
      Karpukhin, Vladimir  and
      Benedict, Brian  and
      McQuade, Mark  and
      Solawetz, Jacob",
    editor = "Dernoncourt, Franck  and
      Preo{\c{t}}iuc-Pietro, Daniel  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-industry.36",
    doi = "10.18653/v1/2024.emnlp-industry.36",
    pages = "477--485",
    abstract = "The rapid growth of open-source language models provides the opportunity to merge model checkpoints, combining their parameters to improve performance and versatility. Advances in transfer learning have led to numerous task-specific models, which model merging can integrate into powerful multitask models without additional training. MergeKit is an open-source library designed to support this process with an efficient and extensible framework suitable for any hardware. It has facilitated the merging of thousands of models, contributing to some of the world{'}s most powerful open-source model checkpoints. The library is accessible at: https://github.com/arcee-ai/mergekit.",
}
```
