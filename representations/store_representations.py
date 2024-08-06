import click
import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import os
import random
from pathlib import Path
from mergekit._data.models_and_datasets import save_model_and_dataset, model_and_dataset_to_index
from typing import List
import gc
import uuid

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
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

def store_representations(model_name, output_dir, dataset_name, batch_size, max_length, dataset_size, dataset_column, dataset_subset):
    # Generate the unique ID using UUID
    unique_id = uuid.uuid4().hex[:4]
    
    #!important: Set seed for consistent batch order across runs
    set_seed(42)
    save_model_and_dataset(model_name, dataset_name)
    model_index, dataset_index = model_and_dataset_to_index(model_name, dataset_name)
    output_name = Path(output_dir) / f'{model_index}_{dataset_index}_id_{unique_id}.h5'.replace("/","_")
    set_seed(42)
    assert not output_name.exists(), f'{output_name} already exists.'
    for reps_name in output_name.parent.iterdir():
        if f'{model_index}_{dataset_index}' in reps_name.name:
            raise ValueError(f'Representations for model {model_index} and dataset {dataset_index} already exist in {output_name.parent}')
    os.makedirs(output_name.parent, exist_ok=True)


    dataset = datasets.load_dataset(dataset_name, split=dataset_subset)
    if dataset_size:
        dataset = dataset.select(range(dataset_size))
    device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() \
            else "cpu"
        
    model = AutoModelForCausalLM.from_pretrained(model_name,  
                                                device_map="auto", 
                                                output_hidden_states=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token


    dataloader = DataLoader(dataset[dataset_column], batch_size=batch_size, shuffle=False, drop_last=True)

    with h5py.File(output_name, 'w') as h5file:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.hidden_states
            
            # Remove the first element to account for the input layer not being considered a model hidden layer
            last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention_mask)[1:]

            last_non_padded_hidden_states = last_non_padded_hidden_states
            for layer, hidden_state in enumerate(last_non_padded_hidden_states):
                layer_group = h5file.require_group(f'layer_{layer:03d}')
                file_name = f'batch_{batch_idx}.pt'
                
                layer_group.create_dataset(file_name, data=hidden_state.to('cpu'), compression="gzip")

            # Ensure that the length of last_non_padded_hidden_states matches the number of model hidden layers minus one
            assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, "Length of last_non_padded_hidden_states  \
            does not match expected number of hidden layers."
    
    if torch.cuda.is_available():
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        gc.collect()

@click.command()
@click.option('--model_name', default="BEE-spoke-data/smol_llama-220M-GQA", help='model to use.')
@click.option('--output_dir', default="./representations/representations_store", help='folder to store the result in.')
@click.option('--dataset_name', default="arcee-ai/sec-data-mini", help='dataset to use.')
@click.option('--batch_size', default=8, help='batch size.')
@click.option('--max_length', default=1024, help='maximum length of the input.')
@click.option('--dataset_size', default=4000, help='size of the dataset.')
@click.option('--dataset_column', default="text", help='column of the dataset to use.')
@click.option('--dataset_subset', default="train", help='subset of the dataset to use.')
def main(model_name, output_dir, dataset_name, batch_size, max_length, dataset_size, dataset_column, dataset_subset):
    store_representations(model_name, output_dir, dataset_name, batch_size, max_length, dataset_size, dataset_column, dataset_subset)
if __name__ == "__main__":
    main()
