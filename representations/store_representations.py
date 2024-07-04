# WORK IN PROGRESS
 
import click
import torch
import yaml

from mergekit.config import MergeConfiguration

import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets

import os

logging.basicConfig(level=logging.INFO)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

import torch
from typing import List

import h5py
import torch
import random
import numpy as np

def load_batch_from_hdf5(model_name, batch_idx):
    with h5py.File('batches.h5', 'r') as h5file:
        dataset_name = f'{model_name}/batch_{batch_idx}'
        batch_data = h5file[dataset_name][:]
        batch_tensor = torch.tensor(batch_data)
    return batch_tensor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states


@click.command()
@click.option('--model_path', default="BEE-spoke-data/smol_llama-220M-GQA", help='model to use.')
@click.option('--output_path', default="./representations/", help='folder to store the result in.')
@click.option('--dataset', default="arcee-ai/sec-data-mini", help='dataset to use.')
@click.option('--batch_size', default=8, help='batch size.')
@click.option('--max_length', default=1024, help='maximum length of the input.')
@click.option('--dataset_size', default=4000, help='size of the dataset.')
@click.option('--dataset_column', default="text", help='column of the dataset to use.')
@click.option('--dataset_subset', default="train", help='subset of the dataset to use.')
def main(model_path, output_path, dataset, batch_size, max_length, dataset_size, dataset_column, dataset_subset):

    device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() \
            else "cpu"

    # if resource is a problem
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16)

    dataset = datasets.load_dataset(dataset, split=dataset_subset)
    if dataset_size:
        dataset = dataset.select(range(dataset_size))

        
    model = AutoModelForCausalLM.from_pretrained(model_path,  
                                                device_map="auto", 
                                                quantization_config=quantization_config if device == "cuda" else None, 
                                                output_hidden_states=True)


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    set_seed(42)

    dataloader = DataLoader(dataset[dataset_column], batch_size=batch_size, shuffle=False, drop_last=True)
    
    output_name = f'NEW_Representations_{model.name_or_path.replace("/","_")}_{dataset_subset}_{dataset_size}'
    assert not os.path.exists(output_path+f'{output_name}.h5'), f'{output_name}.h5 already exists.'

    with h5py.File(f'{output_name}.h5', 'w') as h5file:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.hidden_states
            last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention_mask)

            # Remove the first element to account for the input layer not being considered a model hidden layer
            # This adjustment is necessary for analyses focusing on the model's internal transformations
            last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
            for layer, hidden_state in enumerate(last_non_padded_hidden_states):
                layer_group = h5file.require_group(f'layer_{layer}')
                file_name = f'batch_{batch_idx}.pt'
                
                layer_group.create_dataset(file_name, data=hidden_state.to('cpu'), compression="gzip")

            # Ensure that the length of last_non_padded_hidden_states matches the number of model hidden layers minus one
            assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, "Length of last_non_padded_hidden_states  \
            does not match expected number of hidden layers."


if __name__ == "__main__":
    main()
