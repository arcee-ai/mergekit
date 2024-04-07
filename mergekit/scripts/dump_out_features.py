import itertools
import logging
import os
import sys
from typing import List, Optional

import click
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from mergekit.architecture import (
    ProceduralSpaceInfo,
    WeightInfo,
    _template_substitution,
    get_architecture_info,
)
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options
from mergekit.scripts.zipit_utils import CovarianceMetric, remove_pads

logging.basicConfig(level=logging.INFO)

# set seed
torch.manual_seed(42)
np.random.seed(42)


def parse_items(ctx, param, value):
    if value is not None:
        return [item.strip() for item in value.split(",")]


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
    default="up_${layer_index},attn_v_${layer_index}",
    callback=parse_items,
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

    # TODO: revisit this heuristic
    residual_space = None
    for layer_template in _json.layer_templates.weights:
        if "layer_index" not in layer_template:
            residual_space = layer_template
            break

    if residual_space is None:
        raise ValueError("No residual space found in the architecture definition")

    ignore_spaces = ignore_spaces + [residual_space]

    input_space_to_weight = {}
    for layer_template in _json.layer_templates.weights:
        if layer_template.input_space in ignore_spaces:
            continue
        if layer_template.input_space not in input_space_to_weight:
            input_space_to_weight[layer_template.input_space] = []
        input_space_to_weight[layer_template.input_space].append(layer_template.name)

    output_space_to_weight = {}
    for layer_template in _json.layer_templates.weights:
        if layer_template.output_space in ignore_spaces:
            continue
        if layer_template.output_space not in output_space_to_weight:
            output_space_to_weight[layer_template.output_space] = []
        output_space_to_weight[layer_template.output_space].append(layer_template.name)

    # raison d'etre: we want to avoid duplication of effort (in terms of hook)

    input_counts = {k: len(v) for k, v in input_space_to_weight.items()}
    output_count = {k: len(v) for k, v in output_space_to_weight.items()}
    to_remove = []

    for k, v in input_counts.items():
        if k in output_count:
            if v == 1 and output_count[k] == 1:
                to_remove.append(k)

    # remove keys from output
    output_space_to_weight = {
        k: v for k, v in output_space_to_weight.items() if k not in to_remove
    }

    num_layers = model_arch_info.num_layers(model_config)

    # TODO expand the space_names (i.e fill in the index names)

    i = {}
    for k, v in input_space_to_weight.items():
        for j in range(num_layers):
            f = lambda x: _template_substitution(
                x, num_layers=num_layers, layer_index=j
            )
            i[f(k)] = [f(_v) for _v in v]

    o = {}
    for k, v in output_space_to_weight.items():
        for j in range(num_layers):
            f = lambda x: _template_substitution(
                x, num_layers=num_layers, layer_index=j
            )
            o[f(k)] = [f(_v) for _v in v]

    input_space_to_weight = i
    output_space_to_weight = o

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

    feature_storage = {"running_residual": None, "attention_mask": None}

    storage_dict = {}

    def get_attention_output_hook(storage_dict, space_name, capture_input=True):
        """
        Returns a hook function that stores the output of the attention layer.
        """

        def hook(module, input, output):
            # output is a tuple, where the first element is the attention output
            if capture_input:
                storage_dict[space_name] = input.detach()
            else:
                storage_dict[space_name] = output[0].detach()

        return hook

    for k, v in input_space_to_weight.items():
        for i in v:
            model.get_layer(i).register_forward_hook(
                get_attention_output_hook(feature_storage, k, capture_input=True)
            )
    for k, v in output_space_to_weight.items():
        for i in v:
            model.get_layer(i).register_forward_hook(
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
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

            # Store attention masks
            attention_mask = inputs["attention_mask"]
            if feature_storage["attention_mask"] is None:
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
            if feature_storage[residual_space] is None:
                feature_storage[residual_space] = hidden_states
            else:
                feature_storage[residual_space] = torch.cat(
                    (feature_storage[residual_space], hidden_states), dim=0
                )

            for space_name, v in storage_dict.items():
                if feature_storage[space_name] is None:
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
