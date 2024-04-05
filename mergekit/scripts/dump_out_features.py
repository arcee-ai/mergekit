import logging
import os
from typing import List, Optional

import click
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from mergekit.scripts.zipit_utils import CovarianceMetric, remove_pads

logging.basicConfig(level=logging.INFO)

# set seed
torch.manual_seed(42)
np.random.seed(42)


def parse_items(ctx, param, value):
    if value is not None:
        return [item.strip() for item in value.split(",")]


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
):
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
            if feature_storage["running_residual"] is None:
                feature_storage["running_residual"] = hidden_states
            else:
                feature_storage["running_residual"] = torch.cat(
                    (feature_storage["running_residual"], hidden_states), dim=0
                )

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
