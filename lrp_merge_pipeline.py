#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys

# Config file is auto injected

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # model can be user-specific

# These placeholders will be replaced by absolute paths from the notebook
MODEL_1 = "./models/tinyllama-global-full"
MODEL_2 = "./models/tinyllama-local-full"

OUTPUT_DIR = "./models/merged-model"

# Toggle this if GPU available
USE_CUDA = True

# validation

def validate():
    print("Step 1: Checking local model paths...")

    for path in [MODEL_1, MODEL_2]:
        if os.path.exists(path):
            print(f"Found: {path}")
        else:
            print(f" Missing model: {path}")
            # We won't raise error here to see the full debug output


# YAML GENERATION

def generate_yaml():
    print("Step 2: Generating YAML...")

    yaml = f"""
merge_method: lrp

base_model:
  model: \"{BASE_MODEL}\"

parameters:
  density: 0.7
  use_lrp: true

models:
  - model: \"{MODEL_1}\"
    parameters:
      weight: 1.0

  - model: \"{MODEL_2}\"
    parameters:
      weight: 1.0
"""

    with open("lrp_config.yaml", "w") as f:
        f.write(yaml.strip())

    print("\nYAML Generated:\n")
    print(yaml)

# MERGE EXECUTION

def run_merge():
    print("\nStep 3: Running merge...\n")

    # Use the verified path
    mergekit_exec = "/usr/local/bin/mergekit-yaml"
    if not os.path.exists(mergekit_exec):
        mergekit_exec = shutil.which("mergekit-yaml")

    if not mergekit_exec:
        raise RuntimeError("✗ mergekit-yaml not found. Install mergekit.")

    cmd = [
        mergekit_exec,
        "lrp_config.yaml",
        OUTPUT_DIR,
        "--copy-tokenizer",
        "--allow-crimes",
    ]

    if USE_CUDA:
        cmd.append("--cuda")

    print("Running command:")
    print(" ".join(cmd), "\n")

    # Run and show output in real-time
    res = subprocess.run(cmd)

    if res.returncode != 0:
        raise RuntimeError(f"✗ Merge failed with code {res.returncode}")

    print("\nMerge completed successfully!")
    print(f"📁 Output: {OUTPUT_DIR}")


# MAIN


def main():
    print("=== LRP MERGE PIPELINE START ===\n")

    validate()
    generate_yaml()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_merge()

    print("\n✓ ALL DONE")


if __name__ == "__main__":
    main()
