#!/usr/bin/env python3
# filepath: /fsx/lewis/git/hf/mergekit/merge_scan.sh

import os
import yaml
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

# Define paths
RECIPE_PATH = "recipes/R1-Distill-Qwen-Math-7B/v00.00_v01.00_task-arithmetic.yml"
OUTPUT_DIR = "scratch/v00.00_v01.00_task-arithmetic"

def load_yaml(file_path):
    """Load YAML file and return its content."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(content, file_path):
    """Save content to YAML file."""
    with open(file_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False, sort_keys=False)

def update_weights(yaml_content, weight_1, weight_2):
    """Update weights in the YAML content."""
    yaml_content["models"][0]["parameters"]["weight"] = weight_1
    yaml_content["models"][1]["parameters"]["weight"] = weight_2
    return yaml_content

def update_lambda(yaml_content, new_lambda):
    """Update lambda in the YAML content."""
    yaml_content["parameters"]["lambda"] = new_lambda
    return yaml_content

def run_merge(yaml_path, output_dir):
    """Run mergekit-yaml command."""
    cmd = ["mergekit-yaml", yaml_path, output_dir, "--cuda"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_push_to_hub(output_dir):
    """Run push_to_hub.py command."""
    cmd = ["python", "push_to_hub_task_arithmetic.py", "--path", output_dir]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Load the original YAML file
    original_yaml = load_yaml(RECIPE_PATH)
    
    # Create a temp directory for modified YAML files
    os.makedirs("scratch/temp_yamls_task_arithmetic", exist_ok=True)
    
    # Scan through weight pairs
    for i in np.arange(0, 1.1, 0.1):
        lmda = float(round(i, 1))
        print(f"\n{'='*60}")
        print(f"PROCESSING LAMBDA: {lmda}")
        print(f"{'='*60}\n")
        
        # Create modified YAML file
        modified_yaml = update_lambda(original_yaml.copy(), lmda)
        temp_yaml_path = f"scratch/temp_yamls_task_arithmetic/v00.00_v01.00_lambda_{lmda}.yml"
        save_yaml(modified_yaml, temp_yaml_path)
        
        # Ensure output directory is clean
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        
        # Run merge
        try:
            run_merge(temp_yaml_path, OUTPUT_DIR)
            
            # Push to hub
            run_push_to_hub(OUTPUT_DIR)
            
            print(f"Successfully processed lambda {lmda}")
        except Exception as e:
            print(f"Error processing lambda {lmda}: {e}")
            
        # Add a short pause between iterations
        time.sleep(2)

if __name__ == "__main__":
    main()