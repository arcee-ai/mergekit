import logging
import os
import click
from typing import Optional, List
import datasets
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)

# set seed
torch.manual_seed(42)
np.random.seed(42)

def parse_items(ctx, param, value):
    if value is not None:
        return [item.strip() for item in value.split(",")]

@click.command("mergekit-activations-dump")
@click.argument("model-path", type=str)
@click.option("--dataset", "-d", required=True, type=str, help="Dataset to use for activations")
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option("--batch-size", "-b", type=int, default=2, help="Batch size")
@click.option("--dataset-size", "-s", type=int, default=None, help="Dataset size. If None, use full dataset")
@click.option("--dataset-column", "-c", type=str, default="text", help="Dataset column to use")
@click.option("--dataset-subset", "-u", type=str, default="eval", help="Dataset subset to use")
@click.option("--max-length", "-l", type=int, default=512, help="Max length")
@click.option("--dtype", type=str, default=None, help="Data type to convert weights to")
@click.option("--device", type=str, default=None, help="device to compute the activations")



def main(model_path: str, dataset: str, dataset_column: str, out_path: str, batch_size: int, max_length: int, dataset_size: Optional[int], dataset_subset: Optional[str], dtype: Optional[str], device: Optional[str]):
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    def get_attention_output_hook(storage_dict, layer_name):
        """
        Returns a hook function that stores the output of the attention layer.
        """
        def hook(module, input, output):
            # output is a tuple, where the first element is the attention output
            attention_output = output[0]
            storage_dict[layer_name] = attention_output.detach()
        return hook

    attention_outputs_storage = {}
    # Loop through each layer and register the hook
    for i, layer in enumerate(model.layers):
        layer_name = f"layer_{i}_attention_output"
        hook = get_attention_output_hook(attention_outputs_storage, layer_name)
        # Register the hook on the self_attn layer
        layer.self_attn.register_forward_hook(hook)

    dataset = datasets.load_dataset(dataset)[dataset_subset]

    if dataset_size is not None:
        logging.info("Using dataset size %s", dataset_size)
        dataset = dataset.select(range(dataset_size))
    dataset = dataset[dataset_column]

    datasets_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_storage = {}  # Initialize storage for features and attention masks

    for batch in datasets_dataloader:
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states

            # Store hidden states
            for i, hidden_state in enumerate(hidden_states):
                identifier_hs = f"layer_{i}_hidden_state"
                if identifier_hs not in feature_storage:
                    feature_storage[identifier_hs] = []
                feature_storage[identifier_hs].extend(hidden_state.cpu().detach())

            # Store attention masks
            attention_mask = inputs['attention_mask']
            if 'attention_mask' not in feature_storage:
                feature_storage['attention_mask'] = []
            feature_storage['attention_mask'].extend(attention_mask.cpu().detach())

    # Stack all tensors in feature storage
    for identifier in feature_storage:
        feature_storage[identifier] = torch.stack(feature_storage[identifier], dim=0)
        
    # Add attention output storage
    for identifier, tensor in attention_outputs_storage.items():
        feature_storage[identifier] = tensor
    
    if "/" in model_path:
        model_path = model_path.replace("/", "_")

    save_file(feature_storage, f"{out_path}/{model_path}_features.safetensor")

if __name__ == "__main__":
    main()


# python new_dump_me.py TinyLlama/TinyLlama-1.1B-Chat-v0.6 -o ./dump_output  -d arcee-ai/pmc-test-perplexity  -s 2  -c text  -u test  --device cpu