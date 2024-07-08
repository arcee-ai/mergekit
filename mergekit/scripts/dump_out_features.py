import itertools
import logging
import os
import sys
from collections import defaultdict
from typing import List, Optional

import click
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from mergekit.architecture import _template_substitution, get_architecture_info
from mergekit.common import ModelReference

logging.basicConfig(level=logging.INFO)

# set seed
torch.manual_seed(42)
np.random.seed(42)


def clean_name(name):
    return name.replace(".weight", "").replace("model.", "")


def parse_items(ctx, param, value):
    if value is not None:
        return [item.strip() for item in value.split(",")]


def remove_pads(attention_mask, feature_vector):
    batch_size, seq_length = attention_mask.shape
    if (
        len(feature_vector.shape) == 3
    ):  # Hidden states: (batch_size, seq_length, embedding_dim)
        # Expand mask to match the feature_vector dimensions and apply it
        expanded_mask = attention_mask.unsqueeze(-1)
        filtered_feature_vector = feature_vector * expanded_mask
    elif (
        len(feature_vector.shape) == 4
    ):  # Attention outputs: (batch_size, num_attention_heads, seq_length, seq_length)
        # Expand mask for application to attention outputs
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
        # Apply mask to the "keys" dimension of the attention scores
        filtered_feature_vector = feature_vector * expanded_mask
        # Apply mask to the "queries" dimension of the attention scores (transpose mask application)
        expanded_mask_transposed = attention_mask.unsqueeze(1).unsqueeze(2)
        filtered_feature_vector = filtered_feature_vector * expanded_mask_transposed
    else:
        raise ValueError("Unsupported feature vector shape.")

    return filtered_feature_vector


"""

What this script does:

It tries to map input/output spaces to activation maps

it denotes any input space that is found across layers as a residual stream

"""


@click.command("mergekit-activations-dump")
@click.argument("model-path", type=str)
@click.option(
    "--dataset", "-d", required=True, type=str, help="Dataset to use for activations"
)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option("--batch-size", "-b", type=int, default=2, help="Batch size")
@click.option(
    "--dataset-size",
    "-s",
    type=int,
    default=None,
    help="Dataset size. If None, use full dataset",
)
@click.option(
    "--dataset-column", "-c", type=str, default="text", help="Dataset column to use"
)
@click.option(
    "--dataset-subset", "-u", type=str, default="eval", help="Dataset subset to use"
)
@click.option("--max-length", "-l", type=int, default=512, help="Max length")
@click.option("--dtype", type=str, default=None, help="Data type to convert weights to")
@click.option(
    "--device", type=str, default=None, help="device to compute the activations"
)
@click.option(
    "--ignore-spaces",
    "-i",
    type=str,
    default="",
    callback=parse_items,
    help="Spaces to ignore separated by comma. Example: up_${layer_index}",
)
def main(
    model_path: str,
    dataset: str,
    dataset_column: str,
    out_path: str,
    batch_size: int,
    max_length: int,
    dataset_size: Optional[int],
    dataset_subset: Optional[str],
    dtype: Optional[str],
    device: Optional[str],
    ignore_spaces: Optional[List[str]],
):
    # sorting out locations to hook into
    # we do this via the predefined json architecture definitions in mergekit

    model = ModelReference.model_validate(model_path)

    model_config = model.config()
    model_arch_info = get_architecture_info(model_config)

    # things to do: find the residual space
    # rest difference the ignore_spaces
    # residual space is the one to for hidden states
    # the rest we will attach hooks based on the module name

    _json = model_arch_info.definition

    residual_space = None

    weights = []
    for weight in _json.layer_templates.weights:
        if weight.is_kq:
            residual_space = weight.input_space
        weights.append(weight)

    # TODO: revisit this
    if residual_space is None:
        raise ValueError("No residual space found")

    # just a list of connected components
    input_space_to_weights = defaultdict(list)
    output_space_to_weights = defaultdict(list)

    for layer_template in weights:
        if (
            not layer_template.input_space
            or layer_template.input_space in ignore_spaces
        ):
            continue
        input_space_to_weights[layer_template.input_space].append(layer_template.name)

    for layer_template in weights:
        if (
            not layer_template.output_space
            or layer_template.output_space in ignore_spaces
        ):
            continue
        output_space_to_weights[layer_template.output_space].append(layer_template.name)

    # remove the residual space from the input and output
    input_space_to_weights.pop(residual_space, None)
    output_space_to_weights.pop(residual_space, None)

    # NOTE: if the the input space and output space are the same
    # and they go in from one weight and into another weight
    # we can remove the space from the output
    # as the hook need only be applied to capture input from the input weight
    input_counts = {k: len(v) for k, v in input_space_to_weights.items()}
    output_count = {k: len(v) for k, v in output_space_to_weights.items()}
    to_remove = []

    for k, v in input_counts.items():
        if k in output_count:
            if v == 1 and output_count[k] == 1:
                to_remove.append(k)

    # remove keys from output
    output_space_to_weights = {
        k: v for k, v in output_space_to_weights.items() if k not in to_remove
    }
    # -----------------------------------------------------------------

    num_layers = model_arch_info.num_layers(model_config)

    # TODO expand the space_names (i.e fill in the index names)

    i = {}
    for k, v in input_space_to_weights.items():
        print(k)
        for j in range(num_layers):
            f = lambda x: _template_substitution(x, num_layers=num_layers, layer_idx=j)
            i[f(k)] = [f(_v) for _v in v]

    o = {}
    for k, v in output_space_to_weights.items():
        print(k)
        for j in range(num_layers):
            f = lambda x: _template_substitution(x, num_layers=num_layers, layer_idx=j)
            o[f(k)] = [f(_v) for _v in v]

    input_space_to_weights = i
    output_space_to_weights = o

    model = AutoModel.from_pretrained(
        model_path, output_attentions=True, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    dataset = datasets.load_dataset(dataset)[dataset_subset]

    if dataset_size is not None:
        logging.info("Using dataset size %s", dataset_size)
        dataset = dataset.select(range(dataset_size))
    dataset = dataset[dataset_column]

    datasets_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_storage = {}
    storage_dict = {}

    def get_attention_output_hook(storage_dict, space_name, capture_input=True):
        """
        Returns a hook function that stores the output of the attention layer.
        """

        def hook(module, input, output):
            # NOTE: shape of input is [batch, seq_len, dim] and output is Tuple[(seq_len, dim),...]
            if capture_input:
                o = input[0].detach()
            else:
                o = output.detach()

            if space_name not in storage_dict:
                storage_dict[space_name] = o
            else:
                storage_dict[space_name] = torch.cat(
                    (storage_dict[space_name], o), dim=0
                )

        return hook

    for k, v in input_space_to_weights.items():
        for i in v:
            i = clean_name(i)
            model.get_submodule(i).register_forward_hook(
                get_attention_output_hook(feature_storage, k, capture_input=True)
            )
    for k, v in output_space_to_weights.items():
        for i in v:
            i = clean_name(i)
            print(i)
            model.get_submodule(i).register_forward_hook(
                get_attention_output_hook(feature_storage, k, capture_input=False)
            )

    for batch in datasets_dataloader:
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(
                **inputs, output_hidden_states=True, output_attentions=False
            )

            # NOTE: https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput

            # Store attention masks
            attention_mask = inputs["attention_mask"]
            if "attention_mask" not in feature_storage:
                print(attention_mask.shape)
                feature_storage["attention_mask"] = attention_mask.cpu().detach()
            else:
                feature_storage["attention_mask"] = torch.cat(
                    (feature_storage["attention_mask"], attention_mask.cpu().detach()),
                    dim=0,
                )

            hidden_states = [
                remove_pads(attention_mask, hidden_state)
                for hidden_state in outputs.hidden_states
            ]
            hidden_states = torch.stack(outputs.hidden_states, dim=1)

            # stack them
            if residual_space not in feature_storage:
                feature_storage[residual_space] = hidden_states
            else:
                feature_storage[residual_space] = torch.cat(
                    (feature_storage[residual_space], hidden_states), dim=0
                )

            for space_name, v in storage_dict.items():
                if space_name not in feature_storage:
                    feature_storage[space_name] = v
                else:
                    feature_storage[space_name] = torch.cat(
                        (feature_storage[space_name], v), dim=0
                    )

            storage_dict = {}

    # Stack all tensors in feature storage
    for k, v in feature_storage.items():
        if v is not None:
            print(k, v.shape)

    if "/" in model_path:
        model_path = model_path.replace("/", "_")

    # create output directory
    os.makedirs(out_path, exist_ok=True)

    save_file(feature_storage, f"{out_path}/{model_path}_features.bin")


if __name__ == "__main__":
    main()

#  python dump_out_features.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o ./dump_output  -d arcee-ai/pmc-test-perplexity  -s 2  -c text  -u test  --device cpu
#  python dump_out_features.py TinyLlama/TinyLlama-1.1B-Chat-v0.6 -o ./dump_output  -d arcee-ai/pmc-test-perplexity  -s 2  -c text  -u test  --device cpu
#
#  examine the output
#  python
# import torch
# import safetensors
# from safetensors.torch import load_file
# features = load_file("./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v1.0_features.bin")
# features.keys()
# features['running_residual'].shape
# features['attention_mask'].shape
# features['up_0'].shape
