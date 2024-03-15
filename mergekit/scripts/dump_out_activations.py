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
from typing import DefaultDict, Dict, List, Optional, Set

import click
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo, get_architecture_info
from mergekit.common import ModelReference, dtype_from_name


# taken from DALM
def mean_hidden_state(hidden_state, mask):
    a2 = torch.sum(hidden_state * mask.unsqueeze(-1), 1)
    return a2 / torch.clamp(mask.sum(-1, keepdim=True), min=1e-9)


@click.command("mergekit-activations-dump")
@click.argument("model-path", type=str)
@click.option(
    "--target-model-path",
    "-t",
    required=True,
    type=str,
    help="Target model to align weights to",
)
@click.option(
    "--dataset", "-d", required=True, type=str, help="Dataset to use for activations"
)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option("--batch-size", "-b", type=int, default=2, help="Batch size")
@click.option("--max-length", "-l", type=int, default=512, help="Max length")
@click.option(
    "--dtype",
    type=str,
    default=None,
    help="Data type to convert weights to",
)
def main(
    model_path: str,
    target_model_path: str,
    dataset: str,
    out_path: str,
    batch_size: int,
    max_length: int,
    dtype: Optional[str],
):
    model = ModelReference.model_validate(model_path)
    target_model = ModelReference.model_validate(target_model_path)

    # TODO: make subset commandline configurable
    dataset = datasets.load_dataset(dataset)["eval"]

    model_config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    activations = [
        [
            torch.zeros(model_config.hidden_size)
            for _ in range(model_config.num_hidden_layers + 1)
        ]
        for _ in range(2)
    ]

    for model_index, model in enumerate([model_path, target_model]):
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.output_hidden_states = True
        model_config.return_dict = True

        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)

        model.eval()
        model.to("cuda")

        # TODO set seed somewhere for reproducibility
        datasets_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch in datasets_dataloader:
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                # next token prediction
                outputs = model(**inputs)
                # get mask
                mask = inputs["attention_mask"]

                # lot of loss of information with repeated averaging, maybe just the CLS token?
                hidden_states = [
                    (1.0 / batch_size) * torch.sum(mean_hidden_state(x, mask), dim=0)
                    for x in outputs.hidden_states
                ]
                activations[model_index] = activations[model_index] + hidden_states
