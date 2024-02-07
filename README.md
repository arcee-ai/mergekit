> [!NOTE]
> [Charles Goddard](https://github.com/cg123), author of `mergekit` (and person writing this note), has joined [arcee.ai](https://www.arcee.ai/). I will be continuing to develop `mergekit` with their backing and it will continue to be open source under the same license. For more words about this, see the blog post [here](https://blog.arcee.ai/arcee-and-mergekit-unite/).

# mergekit

`mergekit` is a toolkit for merging pre-trained language models. `mergekit` uses an out-of-core approach to perform unreasonably elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, with more coming as they catch my attention.

Features:

- Supports Llama, Mistral, GPT-NeoX, StableLM, and more
- Many [merge methods](#merge-methods)
- GPU or CPU execution
- Lazy loading of tensors for low memory use
- Interpolated gradients for parameter values (inspired by Gryphe's [BlockMerge_Gradient](https://github.com/Gryphe/BlockMerge_Gradient) script)
- Piecewise assembly of language models from layers ("Frankenmerging")

## Installation

```sh
git clone https://github.com/cg123/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

If the above fails with the error of:

```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`

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

```
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
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

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

### Tokenizer Source

The `tokenizer_source` field of a configuration file determines what tokenizer is used by the merged model. This also effects how embeddings and language model heads are merged.

This functionality is still experimental and may break. Please file an issue if you encounter any issues with it.

Valid values:

- `base`: use the tokenizer from the base model
- `union`: construct a tokenizer with all tokens from all models
- `model:<model_path>`: use the tokenizer from a specific model

If set, mergekit will find a mapping between each model's vocabulary and the output tokenizer. This allows models with different vocabularies or added tokens to be meaningfully merged.

`tokenizer_source` is compatible with all merge methods, but when used `lm_head`/`embed_tokens` will be merged linearly. For two-model merges, the `embed_slerp` parameter can be set to `true` to use SLERP instead.

If the `tokenizer_source` field is not set, mergekit will fall back to its legacy default behavior. The tokenizer for the base model (or first model in the merge, if no base model is specified) will be copied to the output directory. The parameter matrices for `lm_head`/`embed_tokens` will be truncated to the smallest size present in the merge. In _most_ cases this corresponds to using the tokenizer for the base model.

### Examples

Several examples of merge configurations are available in [`examples/`](examples/).

## Merge Methods

A quick overview of the currently supported merge methods:

| Method                                                                                       | `merge_method` value | Multi-Model | Uses base model |
| -------------------------------------------------------------------------------------------- | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                     | `linear`             | ✅          | ❌              |
| SLERP                                                                                        | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                          | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                     | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)            | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                  | `passthrough`        | ❌          | ❌              |

### Linear

The classic merge method - a simple weighted average.

Parameters:

- `weight` - relative (or absolute if `normalize=False`) weighting of a given tensor
- `normalize` - if true, the weights of all models contributing to a tensor will be normalized. Default behavior.

### SLERP

Spherically interpolate the parameters of two models. One must be set as `base_model`.

Parameters:

- `t` - interpolation factor. At `t=0` will return `base_model`, at `t=1` will return the other one.

### [Task Arithmetic](https://arxiv.org/abs/2212.04089)

Computes "task vectors" for each model by subtracting a base model. Merges the task vectors linearly and adds back the base. Works great for models that were fine tuned from a common ancestor. Also a super useful mental framework for several of the more involved merge methods.

Parameters: same as [Linear](#linear)

### [TIES](https://arxiv.org/abs/2306.01708)

Builds on the task arithmetic framework. Resolves interference between models by sparsifying the task vectors and applying a sign consensus algorithm. Allows you to merge a larger number of models and retain more of their strengths.

Parameters: same as [Linear](#linear), plus:

- `density` - fraction of weights in differences from the base model to retain

### [DARE](https://arxiv.org/abs/2311.03099)

In the same vein as TIES, sparsifies task vectors to reduce interference. Differs in that DARE uses random pruning with a novel rescaling to better match performance of the original models. DARE can be used either with the sign consensus algorithm of TIES (`dare_ties`) or without (`dare_linear`).

Parameters: same as [TIES](#ties) for `dare_ties`, or [Linear](#linear) for `dare_linear`

### Passthrough

`passthrough` is a no-op that simply passes input tensors through unmodified. It is meant to be used for layer-stacking type merges where you have only one input model. Useful for frankenmerging.
