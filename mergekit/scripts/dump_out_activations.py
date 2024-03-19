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

    activations = None
    model = None
    if dump_type == "hidden-state":
        # get layer template from model
        activations = [
            torch.zeros(model_config.hidden_size)
            for _ in range(model_config.num_hidden_layers + 1)
        ]
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.output_hidden_states = True
        model_config.return_dict = True
        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)

    # sizes not known in advance
    elif dump_type == "activation":
        activations = DefaultDict[str, List[torch.Tensor]]()
        model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()
    model.to("cuda")

    # TODO set seed somewhere for reproducibility
    if dataset_size is not None:
        logging.info(f"Using dataset size {dataset_size}")
        dataset = dataset[:dataset_size]
    dataset = dataset[dataset_column]

    datasets_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO: possibly wrong and might leads to loads of memory consumption
    # TODO: running average ?
    def activation_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = [output.detach()]
            else:
                # runnign average here?
                activations[name].append(output.detach())

        return hook

    if dump_type == "activation":
        # get module
        for module_str in hook_modules:
            module = model.get_submodule(module_str)
            module.register_forward_hook(activation_hook(module_str))

    for batch in datasets_dataloader:
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast():
            # next token prediction

            outputs = model(**inputs)
            # get mask
            if dump_type == "hidden-state":
                mask = inputs["attention_mask"]

                # lot of loss of information with repeated averaging, maybe just the CLS token?
                hidden_states = [
                    (1.0 / batch_size) * torch.sum(mean_hidden_state(x, mask), dim=0)
                    for x in outputs.hidden_states
                ]

                for i, hidden_state in enumerate(hidden_states):
                    activations[i] += hidden_state.to("cpu")

    # Question: when activations are used, is there a keyword argument to specify the layer?
    #  Yes, there is a keyword argument to specify the layer
    # TODO: save activations / hidden states to file

    if dump_type == "hidden-state":
        temp = {}
        for i, hidden_state in enumerate(hidden_states):
            temp[f"hidden_state_{i}"] = activations[i] / len(dataset)
        activations = temp

    logging.info(f"Saving activations to {out_path}")
    # dump out activations using safe tensors
    save_file(activations, os.path.join(out_path, "activations.safetensors"))


if __name__ == "__main__":
    main()
