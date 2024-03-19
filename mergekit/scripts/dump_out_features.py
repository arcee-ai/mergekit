# Copyright (C) 2024 Arcee + Charles O. Goddard ?
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
from typing import DefaultDict, Dict, List, Optional, Set

import click
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo, get_architecture_info
from mergekit.common import ModelReference, dtype_from_name

logging.basicConfig(level=logging.INFO)

# set seed
torch.manual_seed(42)
np.random.seed(42)

def parse_items(ctx, param, value):
    if value is not None:
        return [item.strip() for item in value.split(",")]


# taken from DALM
def mean_hidden_state(hidden_state, mask):
    a2 = torch.sum(hidden_state * mask.unsqueeze(-1), 1)
    return a2 / torch.clamp(mask.sum(-1, keepdim=True), min=1e-9)


@click.command("mergekit-activations-dump")
# take in single model path or list of model paths


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
@click.option(
    "--dump-type",
    type=click.Choice(["hidden-state", "activation"], case_sensitive=False),
    default="hidden-state",
    help="Choose between hidden-state or activation",
)
@click.option(
    "--hook-modules",
    callback=parse_items,
    help="Specify modules to hook into separated by commas (e.g., --hook-layers a,b,c)",
    default=None,
)
@click.option(
    "--dtype",
    type=str,
    default=None,
    help="Data type to convert weights to",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="device to compute the activations",
)
def main(
    model_path: str,
    dataset: str,
    dataset_column: str,
    out_path: str,
    batch_size: int,
    max_length: int,
    dump_type: str,
    dataset_size: Optional[int],
    dataset_subset: Optional[str],
    hook_modules: Optional[List[str]],
    dtype: Optional[str],
    device: Optional[str],
):
    # NOTES: it seems somewhat doable that you can hook onto activations

    if dump_type == "activation" and hook_modules is None:
        raise ValueError("hook-layers must be specified for activation dump type")

    model = ModelReference.model_validate(model_path)

    dataset = datasets.load_dataset(dataset)[dataset_subset]

    model_config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
      
    model_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)

    model.eval()
    model.to(device)

    if dataset_size is not None:
        logging.info("Using dataset size %s", dataset_size)
        dataset = dataset[:dataset_size]
    dataset = dataset[dataset_column]

    datasets_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_features(model, layer_names, feature_type='activations'):
        """
        Extracts 'activations' or 'hidden-states' from specified layers of a model.

        Parameters:
        - model (torch.nn.Module): Model for feature extraction.
        - layer_names (list): Target layers for 'activations' mode.
        - feature_type (str): 'activations' (default) or 'hidden-states'.
        
        Returns:
        - features (dict): Captured features, empty for 'hidden-states'.
        - hooks (list): Handles for 'activations' hooks, empty for 'hidden-states'.
        """
        features = {}
        hooks = []

        if feature_type == 'activation':
            def hook_fn(module, input, output):
                for name, mod in model.named_modules():
                    if mod == module:
                        features[name] = output.detach()
                        break

            for name, module in model.named_modules():
                if any(layer_name in name for layer_name in layer_names):
                    hooks.append(module.register_forward_hook(hook_fn))
        
        return features, hooks
    
    # Example usage for capturing activations]
    features, hooks = get_features(model, hook_modules, feature_type=dump_type)

    # Define a dictionary for storing features with identifiers
    feature_storage = {}
    for batch in datasets_dataloader:
        # Wrap the model call in torch.no_grad() to avoid computing gradients
        with torch.no_grad():

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs,  output_hidden_states=(dump_type =='hidden-state'))

            if features:  # Activations were requested
                for name, feature in features.items():
                    identifier = f"{name}_activation"
                    if identifier not in feature_storage:
                        feature_storage[identifier] = []

                    feature_data = feature.cpu().detach()

                    # Split the tensor along the batch dimension and extend the list
                    for single_feature_data in feature_data.cpu().detach():
                        feature_storage[identifier].append(single_feature_data)

            if dump_type == 'hidden-state':
                hidden_states = outputs.hidden_states
                for i, hidden_state in enumerate(hidden_states):
                    identifier = f"layer_{i}_hidden_state"
                    if identifier not in feature_storage:
                        feature_storage[identifier] = []
                    # Split the tensor along the batch dimension and extend the list
                    for single_hidden_state in hidden_state.cpu().detach():
                        feature_storage[identifier].append(single_hidden_state)
    
    # After processing the entire dataset, remove the hooks
    for hook in hooks:
        hook.remove()

    # After processing all batches, you may want to convert lists to numpy arrays for convenience
    for identifier in feature_storage.keys():
        feature_storage[identifier] = torch.stack(feature_storage[identifier], dim=0)

    # Save the features to disk
    save_file(feature_storage,  f"features_{dump_type}.bin")


if __name__ == "__main__":
    main()


# python dump_out_activations.py  gpt2 -o dump_output --dump-type activation -d arcee-ai/pmc-test-perplexity  -s 8  -c text  -u test  --device cpu --hook-modules mlp.act,attn.c_proj