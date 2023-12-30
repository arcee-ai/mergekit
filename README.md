# mergekit

`mergekit` is a toolkit for merging pre-trained language models, using a variety of merge methods including TIES, linear, and slerp merging. The toolkit also enables piecewise assembly of a language model from layers.

Run `pip install -e .` to install the package and make the scripts available.
If the above fails with the error of:
```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```
You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`

The script `mergekit-yaml` takes a YAML configuration file defining the operations to perform.

## Configuration

Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. Can be one of 'ties', 'linear', 'slerp', or 'passthrough'.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type for the merging operation.
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall


### Merge Methods

#### **[Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)** (`"ties"`)
Requires a base model.
Parameters:
- `density` - fraction of weights in differences from the base model to retain
- `weight` - relative (or absolute if `normalize=False`) weighting of a given tensor
- `normalize` - if true, the weights of all models contributing to a tensor will be normalized. Default behavior.


#### Linear
Does not require a base model. Takes parameters `weight` and `normalize`, with same definition as above.


#### SLERP
Requires exactly two models, one of which must be the base model. Takes one parameter - `t` - the interpolation factor from the base model to the secondary model.


### Tokenizer Source

The `tokenizer_source` field of a configuration file determines what tokenizer is used by the merged model. This also effects how embeddings and language model heads are merged.

Valid values:

* `base`: use the tokenizer from the base model
* `union`: construct a tokenizer with all tokens from all models
* `model:<model_path>`: use the tokenizer from a specific model

If set, mergekit will find a mapping between each model's vocabulary and the output tokenizer. This allows models with different vocabularies or added tokens to be meaningfully merged.

`tokenizer_source` is compatible with all merge methods, but when used `lm_head`/`embed_tokens` will be merged linearly. For two-model merges, the `embed_slerp` parameter can be set to `true` to use SLERP instead.

If the `tokenizer_source` field is not set, mergekit will fall back to its legacy default behavior. The tokenizer for the base model (or first model in the merge, if no base model is specified) will be copied to the output directory. The parameter matrices for `lm_head`/`embed_tokens` will be truncated to the smallest size present in the merge. In *most* cases this corresponds to using the tokenizer for the base model.

### Examples

- Simple linear merge of multiple models:

  ```yml
  models:
    - model: psmathur/orca_mini_v3_13b
      parameters:
        weight: 1.0
    - model: WizardLM/WizardLM-13B-V1.2
      parameters:
        weight: 0.3
    - model: garage-bAInd/Platypus2-13B
      parameters:
        weight: 0.5
  merge_method: linear
  dtype: float16
  ```

- `bakllama.py` style layer recombination:

  ```yml
  slices:
    - sources:
      - model: psmathur/orca_mini_v3_13b
        layer_range: [0, 24]
    - sources:
      - model: garage-bAInd/Platypus2-13B
        layer_range: [20, 40]
  merge_method: passthrough
  dtype: float16
  ```

- Gradient SLERP with different weights for mlp/self attention:

  ```yml
  slices:
    - sources:
        - model: psmathur/orca_mini_v3_13b
          layer_range: [0, 40]
        - model: garage-bAInd/Platypus2-13B
          layer_range: [0, 40]
  merge_method: slerp
  base_model: psmathur/orca_mini_v3_13b
  parameters:
    t:
      - filter: self_attn
        value: [0, 0.5, 0.3, 0.7, 1]
      - filter: mlp
        value: [1, 0.5, 0.7, 0.3, 0]
      - value: 0.5 # fallback for rest of tensors
  dtype: float16
  ```

#### Usage

Once you have created the YAML configuration file, run `mergekit-yaml` with the config file and output path as arguments:

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda]
```

## Legacy Wrappers

Mergekit originally featured two separate scripts with different inputs. The functionality of these is maintained in the `mergekit-legacy` and `bakllama` wrappers. Example usage:

```sh
mergekit-legacy ./output-model --base-model TheBloke/Llama-2-13B-fp16 --cuda \
    --merge WizardLM/WizardLM-13B-V1.2 --weight 0.3 --density 0.5 \
    --merge garage-bAInd/Platypus2-13B --weight 0.5 --density 0.5
```

`mergekit-legacy` can output a YAML configuration for easy migration with the `--print-yaml` option.
