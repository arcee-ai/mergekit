import logging
import os
from collections import defaultdict
from typing import List, Optional

import click
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DefaultDataCollator

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
    if (
        len(feature_vector.shape) == 3
    ):  # Hidden states: (batch_size, seq_length, embedding_dim)
        # Expand mask to match the feature_vector dimensions and apply it
        expanded_mask = attention_mask.unsqueeze(-1)
        filtered_feature_vector = feature_vector * expanded_mask
    else:
        raise ValueError("Unsupported feature vector shape.")

    return filtered_feature_vector


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
            storage_dict[space_name] = torch.cat((storage_dict[space_name], o), dim=0)

    return hook


"""

What this script does:

It tries to map input/output spaces to activation maps

"""


@click.command("mergekit-abm-extract-activations")
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
@click.option(
    "--chat-template/--no-chat-template",
    default=False,
    help="use Chat template for inference",
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
    chat_template: Optional[bool],
    dtype: Optional[str],
    device: Optional[str],
    ignore_spaces: Optional[List[str]],
):
    # sorting out locations to hook into
    # we do this via the predefined json architecture definitions in mergekit

    model = ModelReference.model_validate(model_path)

    model_config = model.config()
    model_arch_info = get_architecture_info(model_config)

    _json = model_arch_info.definition

    residual_space = None

    weights = []
    for weight in _json.layer_templates.weights:
        if weight.is_kq:
            residual_space = weight.input_space
        weights.append(weight)

    if residual_space is None:
        raise ValueError("No residual space found")

    # ======================== Mapping spaces to weights ========================

    # just a list of connected components
    space_to_output_weight_templates = defaultdict(list)
    space_to_input_weight_templates = defaultdict(list)

    for layer_template in weights:
        if (
            not layer_template.input_space
            or layer_template.input_space in ignore_spaces
        ):
            continue
        space_to_output_weight_templates[layer_template.input_space].append(
            layer_template.name
        )

    for layer_template in weights:
        if (
            not layer_template.output_space
            or layer_template.output_space in ignore_spaces
        ):
            continue
        space_to_input_weight_templates[layer_template.output_space].append(
            layer_template.name
        )

    # remove the residual space from the input and output
    space_to_input_weight_templates.pop(residual_space, None)
    space_to_output_weight_templates.pop(residual_space, None)

    # NOTE: if space has input and output weights, remove one or the other because hooking
    # into both will result in duplicate activations
    to_remove = []
    for space, input_weights in space_to_input_weight_templates.items():
        if space in space_to_output_weight_templates:
            # if count of input weights and output weights is non zero, remove the space from space to output_weights
            if (
                len(input_weights) > 0
                and len(space_to_output_weight_templates[space]) > 0
            ):
                to_remove.append(space)

    # remove keys from output
    space_to_output_weight_templates = {
        k: v for k, v in space_to_output_weight_templates.items() if k not in to_remove
    }

    num_layers = model_arch_info.num_layers(model_config)

    space_to_input_weights = {}
    for k, v in space_to_input_weight_templates.items():
        for j in range(num_layers):
            f = lambda x: _template_substitution(x, num_layers=num_layers, layer_idx=j)
            space_to_input_weights[f(k)] = [f(_v) for _v in v]

    space_to_output_weights = {}
    for k, v in space_to_output_weight_templates.items():
        for j in range(num_layers):
            f = lambda x: _template_substitution(x, num_layers=num_layers, layer_idx=j)
            space_to_output_weights[f(k)] = [f(_v) for _v in v]

    # ================== Load model, tokenizer for inference and prepare dataset ==================

    model = AutoModel.from_pretrained(
        model_path, output_attentions=True, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_function = None
    if chat_template:
        logging.info("Using chat template for inference")
        tokenize_function = lambda x: tokenizer.apply_chat_template(
            x,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_dict=True,
        )
    else:
        logging.info("Using default tokenizer (no chat template) for inference")
        tokenize_function = lambda x: tokenizer(
            x,
            padding="longest",
            max_length=max_length,
            truncation=True,
        )

    model.eval()
    model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    dataset = datasets.load_dataset(dataset)[dataset_subset]

    if dataset_size is not None:
        logging.info("Using dataset size %s", dataset_size)
        dataset = dataset.select(range(dataset_size))

    def tokenize(element):
        outputs = tokenize_function(element[dataset_column])
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    dataset = dataset.map(tokenize).select_columns(["input_ids", "attention_mask"])

    datasets_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=DefaultDataCollator()
    )

    feature_storage = {}
    storage_dict = {}

    # ================== Hooking into the model ==================

    # NOTE: if the capture input set to True seems confusing, a space's output is a weight recieving input from the space
    for k, v in space_to_output_weights.items():
        for weight in v:
            weight = clean_name(weight)
            model.get_submodule(weight).register_forward_hook(
                get_attention_output_hook(feature_storage, k, capture_input=True)
            )
    for k, v in space_to_input_weights.items():
        for weight in v:
            weight = clean_name(weight)
            model.get_submodule(weight).register_forward_hook(
                get_attention_output_hook(feature_storage, k, capture_input=False)
            )

    # ================== Inference ==================

    for batch in datasets_dataloader:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                **inputs, output_hidden_states=True, output_attentions=False
            )

            # NOTE: https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput

            # Store attention masks
            attention_mask = inputs["attention_mask"]
            if "attention_mask" not in feature_storage:
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

    # ================== Save activations/features ==================

    logging.info("Feature storage:")
    for k, v in feature_storage.items():
        if v is not None:
            logging.info(f"{k}: Shape: {v.shape}")

    abs_path = os.path.abspath(model_path)
    if os.path.exists(abs_path):
        model_path = abs_path

    model_path = model_path.replace("/", "_")

    # create output directory
    os.makedirs(out_path, exist_ok=True)

    save_file(
        feature_storage, os.path.join(out_path, f"{model_path}_features.safetensor")
    )


if __name__ == "__main__":
    main()
