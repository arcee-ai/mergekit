import logging
import os
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import click
import numpy as np
import safetensors
import torch
import tqdm

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo, get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options


@click.command("zipit")
@click.argument("model_path", type=str)
@click.argument("secondary_model_path", type=str)
@click.argument("merge_unmerge_directory", type=str)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option(
    "--dtype",
    type=str,
    default="fp16",
    help="Data type to convert weights to",
)
@add_merge_options
def main(
    model_path: str,
    secondary_model_path,
    merge_unmerge_directory: str,
    out_path: str,
    dtype: Optional[str],
    merge_options: MergeOptions,
):
    model = ModelReference.model_validate(model_path)
    secondary_model = ModelReference.model_validate(secondary_model_path)

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.lora_cache_dir = merge_options.lora_merge_cache
    cache.hf_cache_dir = merge_options.transformers_cache

    for m in tqdm.tqdm([model, secondary_model], desc="Preparing models"):
        print(type(m))
        cache.get(m)

    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    model_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=merge_options.trust_remote_code)
    )

    loader_1 = cache.get(model)
    loader_2 = cache.get(secondary_model)

    # create output directory if doesn't exist
    os.makedirs(out_path, exist_ok=True)

    filtered = ["running_residual"]
    merge_unmerge_dictionary = {}
    for i in filtered:
        m = safetensors.torch.load(
            os.path.join(merge_unmerge_directory, f"{i}_merge.safetensor")
        )
        u = safetensors.torch.load(
            os.path.join(merge_unmerge_directory, f"{i}_unmerge.safetensor")
        )
        merge_unmerge_dictionary[i] = (m["running_residual"], u["running_residual"])

    # TODO: deal with the embedding matrix on both ends

    # the place where both models are aligned and saved
    for weight_info in model_arch_info.all_weights(config=model_config):
        # upstream merge , downstream unmerge

        merge_matrix, unmerge_matrix = None, None

        if weight_info.input_space in merge_unmerge_dictionary:
            _, unmerge_matrix = merge_unmerge_dictionary[weight_info.input_space]
            # split them into two
            unmerge_matrix = unmerge_matrix.chunk(2, dim=0)

        if weight_info.output_space in merge_unmerge_dictionary:
            merge_matrix, _ = merge_unmerge_dictionary[weight_info.output_space]
            merge_matrix = merge_matrix.chunk(2, dim=1)

        original_w = loader_1.get_tensor(weight_info.name, device="cuda")
        original_w2 = loader_2.get_tensor(weight_info.name, device="cuda")
        if dtype is not None:
            w = original_w.to(dtype=dtype)
            w2 = original_w2.to(dtype=dtype)

        if merge_matrix is not None:
            if weight_info.is_embed:
                w = (merge_matrix[0] @ w.T).T
                w2 = (merge_matrix[1] @ w2.T).T
            else:
                w = merge_matrix[0] @ w
                w2 = merge_matrix[1] @ w2

        if unmerge_matrix is not None:
            w = w @ unmerge_matrix[0]
            w2 = w2 @ unmerge_matrix[1]

        # check if weights have not mutated, if yes then  shoot warning
        if torch.allclose(original_w, w):
            logging.warning(
                f"Weight {weight_info.name} for model 1 has NOT mutated during merge"
            )

        if torch.allclose(original_w2, w2):
            logging.warning(
                f"Weight {weight_info.name} has model 2 has NOT mutated during merge"
            )

        # average weights and save them
        w = (w + w2) / 2
        writer.write_tensor(weight_info.name, w)


#  python mergekit/scripts/zip_fast_prototype.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama/TinyLlama-1.1B-Chat-v0.6  m_v_out -o new_model
main()
