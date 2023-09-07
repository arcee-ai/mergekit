## mergekit

`mergekit` is a toolkit for merging pre-trained language models, using a variety of merge methods including TIES, linear, and slerp merging. The toolkit also enables piecewise assembly of a language model from layers selected from other models using `bakllama.py`.

### Merging Models with `main.py`

#### Usage

To merge models using the `main.py` script, specify the output directory for the final model and the models to be merged using the `--merge` option. Depending on the merge method chosen, other parameters such as `--density`, `--weight`, and `--base-model` might be necessary.

The script supports the following merge methods:

- **[Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)** (default method, 'ties')
  - Requires a base model.
  - Can specify per-model weights and densities.
- **Linear**
  - Does not require a base model.
  - Must specify weights for all models being merged.
- **SLERP**
  - Requires exactly two models.
  - Must specify a single weight to set the interpolation parameter between the two models.

#### Examples

- Merging with TIES method and specifying per-model weights and densities:

  ```sh
  python main.py ./output-model --base-model TheBloke/Llama-2-13B-fp16 --cuda \
      --merge WizardLM/WizardLM-13B-V1.2 --weight 0.3 --density 0.5 \
      --merge garage-bAInd/Platypus2-13B --weight 0.5 --density 0.5
  ```

- Merging with linear method and setting model weights:

  ```sh
  python main.py ./output-model --cuda --method linear \
      --merge garage-bAInd/Platypus2-13B --weight 0.6 \
      --merge WizardLM/WizardLM-13B-V1.2 --weight 0.2
  ```

- Merging with SLERP method and setting interpolation parameter:

  ```sh
  python main.py ./output-model --cuda --method slerp --base-model garage-bAInd/Platypus2-13B \
      --merge WizardLM/WizardLM-13B-V1.2 --weight 0.5
  ```


Refer to the script's help message (`python main.py --help`) for detailed information on all available options.

### Piecewise layer combinations with `bakllama.py`

The `bakllama.py` script allows you to assemble a model piecewise with layers taken from other pre-trained models.
Configuration

To use the bakllama.py script, you need to create a YAML configuration file where you define the layers to be used from various source models, and optionally specify the sources for the embedding and LM head components.

The configuration file should have the following fields:

 - `layer_slices`: A list of layer slice objects, each specifying a range of layers to take from a source model.
   - `model`: The identifier or path of the source model.
   - `start`: The starting layer index (inclusive).
   - `end`: The ending layer index (exclusive).
   - `scale`: (Optional) A scaling factor for the weights of the layers.
 - `embedding_source`: (Optional) The model to take the embedding layer from. If not specified, it defaults to the first model listed in layer_slices.
 - `lm_head_source`: (Optional) The model to take the LM head from. If not specified, it defaults to the last model listed in layer_slices.

#### Usage

Once you have created the YAML configuration file, run `bakllama.py` script with the config file and output path as arguments:

```sh
python bakllama.py path/to/your/config.yml ./output-model-directory
```
