# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
import logging
import shutil
from pathlib import Path

import click
import torch
from safetensors import safe_open
from tqdm import tqdm

from mergekit.architecture import ParameterNamesUtils
from mergekit.io.lazy_tensor_loader import ShardedTensorIndex
from mergekit.io.tensor_writer import TensorWriter

DEFAULT_SHARD_SIZE = 5 * 1024**3


def load_tensor_from_file(tensor_name: str, tensor_file: str = None) -> torch.Tensor:
    """
    Load a specific tensor from a .safetensors file.

    :param tensor_name: The name of the tensor to load.
    :param tensor_file: The .safetensors file that contains the tensor.
    :return: The loaded tensor as a PyTorch tensor.
    """
    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        if tensor_name in f.keys():
            return f.get_tensor(tensor_name)
        else:
            raise ValueError(
                f"Tensor '{tensor_name}' not found in file '{tensor_file}'"
            )


def load_tensor_from_index(tensor_name: str, index: ShardedTensorIndex) -> torch.Tensor:
    """
    Load a specific tensor from a ShardedTensorIndex.

    :param tensor_name: The name of the tensor to load.
    :param index: The ShardedTensorIndex containing the tensor.
    :return: The loaded tensor as a PyTorch tensor.
    """
    return load_tensor_from_file(
        tensor_name, Path(index.base_path) / index.tensor_paths[tensor_name]
    )


def copy_and_fill_missing_params(
    base_model_repo_id: str,
    sub_model_dir: str,
    max_shard_size: int = DEFAULT_SHARD_SIZE,
    output_dir: str = None,
):
    """
    Merge submodel weights into a base model and fill in missing parameters.

    Use Case:
    Given a submodel (e.g., a language model) that is structurally identical to a subset of a
    larger base model (e.g., a vision-language model).
    The submodel contains only a subset of the weights (e.g., for the language model part),
    while the base model contains all weights required for the complete architecture.

    This function replaces the shared parameters in the base model with those from the submodel,
    fascilitating testing after generating submodel parameters through merging.



    Parameters:
        base_model_repo_id (str):
            The path to the base model's directory or its Hugging Face repository ID.
            This model provides all parameters and files required for the complete model.
        sub_model_dir (str):
            The path to the submodel's directory containing the merged weights.
            Parameters in this directory replace the corresponding weights in the base model.
        max_shard_size (int, optional):
            The maximum shard size for saving model weights, in bytes. Defaults to 5 GiB.
        output_dir (str, optional):
            The directory to save the final merged model. If not provided, a default directory
            is created using the names of the base and submodel.

    Returns:
        pathlib.Path:
            The path to the directory where the final merged model is saved.

    Raises:
        AssertionError:
            If the base model has fewer parameters than the submodel, ensuring compatibility.
        ValueError:
            If tensor loading or parameter alignment issues occur.

    Notes:
        - The function does not modify the original base or submodel directories.
        - For Hugging Face repository IDs, ensure the `HF_HOME` environment variable is properly configured.
        - Non-shared parameters, as well as any additional configuration files, are copied from the base model to create a fully functional model.
    """
    # Prepare paths and configurations
    output_dir = (
        Path(sub_model_dir).parent
        / f"{Path(base_model_repo_id).stem}--{Path(sub_model_dir).stem}"
        if output_dir is None
        else Path(output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the model directory for the base model
    base_dir = ParameterNamesUtils.resolve_model_directory(base_model_repo_id)
    files_to_copy = [
        item
        for item in base_dir.rglob("*")
        if item.is_file() and item.suffix not in {".safetensors", ".bin"}
    ]

    # Copy non-parameter files from the base model
    with tqdm(
        total=len(files_to_copy), desc="Copying non-parameter files", unit="file"
    ) as pbar:
        for item in files_to_copy:
            target_path = output_dir / item.relative_to(base_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)
            pbar.update(1)

    # Retrieve parameter names from both models
    base_param_names = ParameterNamesUtils.get_model_parameter_names(base_model_repo_id)
    submodel_param_names = ParameterNamesUtils.get_model_parameter_names(sub_model_dir)

    # Ensure the base model has more parameters than the submodel
    assert len(base_param_names) > len(submodel_param_names), (
        f"Base model must have more parameters than the submodel. "
        f"Base: {len(base_param_names)}, Submodel: {len(submodel_param_names)}"
    )

    # Determine parameter prefix and find common names
    prefix = ParameterNamesUtils.find_prefix(base_param_names, submodel_param_names)
    common_param_names = ParameterNamesUtils.find_common_ordered_names(
        [base_param_names, submodel_param_names], ["", prefix]
    )

    # Load parameter indices for tensor storage
    base_index = ShardedTensorIndex.from_disk(str(base_dir))
    submodel_index = ShardedTensorIndex.from_disk(
        str(ParameterNamesUtils.resolve_model_directory(sub_model_dir))
    )

    # Initialize the tensor writer
    writer = TensorWriter(
        out_path=str(output_dir), max_shard_size=max_shard_size, safe_serialization=True
    )

    # Copy and fill parameters from base to submodel
    for name, tensor_path in tqdm(
        base_index.tensor_paths.items(),
        total=len(base_index.tensor_paths),
        desc="Merging tensors",
        unit="tensor",
    ):
        tensor = load_tensor_from_index(name, base_index)

        # Check if the parameter is common to both models
        if name in common_param_names:
            submodel_name = ParameterNamesUtils.strip_prefix(name, prefix)
            submodel_tensor = load_tensor_from_index(submodel_name, submodel_index)

            # Log size mismatches
            if submodel_tensor.size() != tensor.size():
                logging.warning(
                    f"Size mismatch for tensor '{name}': {tensor.size()} vs {submodel_tensor.size()}"
                )

            tensor = submodel_tensor

        # Save the tensor to the output directory
        writer.save_tensor(name, tensor.clone())

    # Finalize the writer to ensure data is saved and index file is created
    writer.finalize()

    return output_dir


@click.command()
@click.argument("base_model_repo_id", type=str)
@click.argument("sub_model_dir", type=str)
@click.option("--max_shard_size", type=int, default=DEFAULT_SHARD_SIZE)
@click.option("--output_dir", type=str, default=None)
def main(
    base_model_repo_id,
    sub_model_dir,
    max_shard_size,
    output_dir,
):
    copy_and_fill_missing_params(
        base_model_repo_id, sub_model_dir, max_shard_size, output_dir
    )


if __name__ == "__main__":
    main()
