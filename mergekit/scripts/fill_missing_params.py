# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import click
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

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


class ParameterNamesUtils:
    """Utility functions for handling parameter names."""

    @staticmethod
    def resolve_model_directory(repo_id: str) -> Path:
        """Resolve the model directory (local or Hugging Face Hub)."""
        if Path(repo_id).is_dir():
            return Path(repo_id)

        return Path(snapshot_download(repo_id))

    @staticmethod
    def get_model_parameter_names(repo_id: str) -> List[str]:
        """Get parameter names of a model from a Hugging Face repo or local directory."""
        model_dir = ParameterNamesUtils.resolve_model_directory(repo_id)
        return list(ShardedTensorIndex.from_disk(str(model_dir)).tensor_paths.keys())

    @staticmethod
    def strip_prefix(name: str, prefix: str) -> str:
        """Remove a single prefix from the start of a name."""
        if prefix != "" and name.startswith(prefix + "."):
            return name[len(prefix) + 1 :]
        return name

    @staticmethod
    def find_prefix(list1: List[str], list2: List[str]) -> Optional[str]:
        """
        Find a prefix in list1 that, after removal, makes list2 an ordered sublist.
        """
        assert len(list1) >= len(list2), "params name list1 can't be shorter than list2"

        possible_prefixes = {item.split(".")[0] for item in list1 if "." in item}
        possible_prefixes = [""] + list(possible_prefixes)

        prefix_matches = {}
        best_prefix = ""  # Default to no prefix
        for prefix in possible_prefixes:
            stripped_list1 = [
                ParameterNamesUtils.strip_prefix(item, prefix) for item in list1
            ]
            prefix_matches[prefix] = len(
                [item for item in list2 if item in stripped_list1]
            )

        if max(prefix_matches.values()) > prefix_matches[""]:
            best_prefix = max(prefix_matches, key=prefix_matches.get)

        return best_prefix

    @staticmethod
    def find_common_ordered_names(
        param_names: List[List[str]], prefixes: List[str]
    ) -> List[str]:
        """Identify and return common parameter names across all models, ensuring correct order. Also account for prefix."""
        common_names = set(param_names[0])
        for i in range(1, len(param_names)):
            prefix = f"{prefixes[i]}." if prefixes[i] else ""
            common_names.intersection_update({prefix + name for name in param_names[i]})
        return [name for name in param_names[0] if name in common_names]

    @staticmethod
    def remove_size_conflicts(common_names, referenced_models, prefixes):
        model_dirs = [
            ParameterNamesUtils.resolve_model_directory(m.model.path)
            for m in referenced_models
        ]
        model_indices = [ShardedTensorIndex.from_disk(str(dir)) for dir in model_dirs]

        common_name_and_shape = common_names.copy()
        removed_names = []

        for name in common_names:
            base_shape = ParameterNamesUtils.tensor_shape(name, model_indices[0])

            for i in range(1, len(referenced_models)):
                other_name = name
                prefix = f"{prefixes[i]}." if prefixes[i] else ""
                if name.startswith(prefix) and prefix != "":
                    other_name = name[len(prefix) :]
                shape = ParameterNamesUtils.tensor_shape(other_name, model_indices[i])

                if base_shape != shape:
                    common_name_and_shape.remove(name)
                    removed_names.append((name, base_shape, shape, i))
                    break

        size_mismatch_count = len(removed_names)
        if size_mismatch_count > 0:
            logging.warning(
                f"Size mismatch detected for {size_mismatch_count}/{size_mismatch_count + len(common_names)} tensors. "
                "These names were removed from the merge list."
            )
            logging.info(
                "The following tensors have different shapes across models and were removed from the merge list:"
            )
            for name, base_shape, shape, i in removed_names:
                logging.info(
                    f"Tensor name: {name}, Base model shape: {base_shape}, Mismatched shape: {shape} in model {referenced_models[i].model.path}"
                )

        return common_name_and_shape

    @staticmethod
    def are_common_params_ordered(list1: List[str], list2: List[str]) -> bool:
        """
        Check if common elements of list2 maintain their relative order in list1.
        """
        common_params = set(list1).intersection(set(list2))
        last_index = -1

        for param in list2:
            if param in common_params:
                current_index = list1.index(param)
                if current_index < last_index:
                    return False
                last_index = current_index
        return True

    @staticmethod
    def ordered_sublist(list1: List[str], list2: List[str]) -> bool:
        """
        Check if list2 is a contiguous ordered sublist of list1.
        """
        n, m = len(list1), len(list2)

        for i in range(n - m + 1):
            if list1[i : i + m] == list2:
                return True
        return False

    @staticmethod
    def report_names_similarity(
        base_names: List[str], other_names: List[str]
    ) -> Tuple[Optional[str], str]:
        """
        Analyze similarity between parameter names of two models and identify shared prefixes.
        Returns:
            best_prefix (str): Best matching prefix for parameter names.
            case_message (str): Explanation of the structural relationship.
        """
        possible_prefixes = {""}
        possible_prefixes.update(
            {item.split(".")[0] for item in base_names if "." in item}
        )

        prefixes_subset_overlap = {}
        best_prefix = None
        case_message = "No common parameter names found for any prefix"

        for prefix in possible_prefixes:
            base_names_stripped = [
                ParameterNamesUtils.strip_prefix(name, prefix) for name in base_names
            ]

            if ParameterNamesUtils.ordered_sublist(base_names_stripped, other_names):
                return prefix, "All params in model have exact match in base model."

            intersection = set(base_names_stripped).intersection(set(other_names))
            prefixes_subset_overlap[prefix] = intersection

        if prefixes_subset_overlap:
            best_prefix = max(
                prefixes_subset_overlap, key=lambda x: len(prefixes_subset_overlap[x])
            )
            base_names_stripped = [
                ParameterNamesUtils.strip_prefix(name, best_prefix)
                for name in base_names
            ]

            overlap = len(prefixes_subset_overlap[best_prefix])
            ordered = ParameterNamesUtils.are_common_params_ordered(
                base_names_stripped, other_names
            )
            mismatched = [
                item for item in other_names if item not in base_names_stripped
            ]
            mismatched = "\n    ".join(mismatched)
            case_message = (
                f"{overlap}/{len(other_names)} ({100 * overlap / len(other_names):.2f}%) "
                f"of model parameters are in the base model. \n"
                f"  Name ordering is {'preserved' if ordered else 'not preserved'}.\n"
                f"  Missing parameters:\n    {mismatched}"
            )

        return best_prefix, case_message

    @staticmethod
    def tensor_shape(name, index) -> Tuple[int]:
        from safetensors import safe_open

        with safe_open(
            Path(index.base_path) / index.tensor_paths[name], framework="pt"
        ) as f:
            return f.get_slice(name).get_shape()
