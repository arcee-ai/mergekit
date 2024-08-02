import json
from pathlib import Path

def save_model_and_dataset(model_name, dataset_name):
    models_and_datasets = presets().load()
    if model_name not in models_and_datasets['model_names']:
        models_and_datasets['model_names'].append(model_name)
    if dataset_name not in models_and_datasets['dataset_names']:
        models_and_datasets['dataset_names'].append(dataset_name)
    presets().save(models_and_datasets['model_names'], models_and_datasets['dataset_names'])

def model_and_dataset_to_index(model_name, dataset_name):
    models_and_datasets = presets().load()
    model_index = models_and_datasets['model_names'].index(model_name) if model_name in models_and_datasets['model_names'] else []
    dataset_index = models_and_datasets['dataset_names'].index(dataset_name) if dataset_name in models_and_datasets['dataset_names'] else []

    return model_index, dataset_index

def index_to_model_and_dataset(model_index, dataset_index):
    models_and_datasets = presets().load()
    model_name = models_and_datasets['model_names'][model_index] if len(models_and_datasets['model_names']) > model_index else []
    dataset_name = models_and_datasets['dataset_names'][dataset_index] if len(models_and_datasets['dataset_names']) > dataset_index else []
    return model_name, dataset_name

class presets():
    def __init__(self):
        self.FILE_PATH = Path(__file__).parent / 'models_and_datasets.json'

    def load(self):
        """Load the lists from the JSON file."""
        if self.FILE_PATH.exists():
            with open(self.FILE_PATH, 'r') as file:
                data = json.load(file)
                return data
        print(f"File {self.FILE_PATH} does not exist or is empty.")
        return {}

    def save(self, model_names, dataset_names):
        """Save the lists to the JSON file."""
        data = {
            'model_names': model_names,
            'dataset_names': dataset_names
        }
        with open(self.FILE_PATH, 'w') as file:
            json.dump(data, file, indent=4)
