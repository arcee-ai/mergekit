import logging
import os
from typing import Optional

import click
import safetensors.torch
import torch
import tqdm
from transformers import AutoTokenizer

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-activation-based-align")
@click.argument("secondary_model_path", type=str)
@click.argument("merge_unmerge_directory", type=str)
@click.option(
    "--out-path", "-o", required=True, type=str, help="Path to save the aligned model"
)
@click.option(
    "--dtype",
    type=str,
    default="float16",
    help="Data type to convert weights to",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda",
    help="Device to compute on (default: cuda)",
)
@add_merge_options
def main(
    secondary_model_path,
    merge_unmerge_directory: str,
    out_path: str,
    dtype: Optional[str],
    device: Optional[str],
    merge_options: MergeOptions,
):
    secondary_model = ModelReference.model_validate(secondary_model_path)

    dtype = dtype_from_name(dtype) if dtype else None

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.hf_cache_dir = merge_options.transformers_cache

    cache.get(secondary_model)

    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    model_config = secondary_model.config(
        trust_remote_code=merge_options.trust_remote_code
    )
    model_arch_info = get_architecture_info(
        secondary_model.config(trust_remote_code=merge_options.trust_remote_code)
    )

    loader = cache.get(secondary_model)

    os.makedirs(out_path, exist_ok=True)

    merge_unmerge_dictionary = {}

    spaces = [
        f.split("_unmerge")[0]
        for f in os.listdir(merge_unmerge_directory)
        if "_unmerge" in f
    ]
    for i in spaces:
        logging.info(f"Loading merge/unmerge tensors for {i}")
        m = safetensors.torch.load_file(
            os.path.join(merge_unmerge_directory, f"{i}_merge.safetensor"),
            device=device,
        )
        u = safetensors.torch.load_file(
            os.path.join(merge_unmerge_directory, f"{i}_unmerge.safetensor"),
            device=device,
        )
        merge_unmerge_dictionary[i] = (
            m[i].to(device, dtype=dtype),
            u[i].to(device, dtype=dtype),
        )

    for weight_info in model_arch_info.all_weights(config=model_config):
        merge_matrix, unmerge_matrix = None, None

        if weight_info.input_space in merge_unmerge_dictionary:
            _, unmerge_matrix = merge_unmerge_dictionary[weight_info.input_space]

        if weight_info.output_space in merge_unmerge_dictionary:
            merge_matrix, _ = merge_unmerge_dictionary[weight_info.output_space]

        original_w = loader.get_tensor(weight_info.name, device=device)

        if dtype is not None:
            original_w = original_w.to(dtype=dtype)
            original_w2 = original_w2.to(dtype=dtype)

        w = torch.clone(original_w)

        if not merge_matrix and not unmerge_matrix:
            logging.warning(
                f"❌ Weight {weight_info.name} for model has no merge or unmerge matrix !!"
            )

        if merge_matrix is not None:
            if weight_info.is_embed:
                w = w @ merge_matrix[0].T
            else:
                w = merge_matrix[0] @ w

        if unmerge_matrix is not None:
            w = w @ unmerge_matrix[0]

        if torch.allclose(original_w, w):
            logging.warning(
                f"❌ Weight {weight_info.name} for input model has NOT mutated during merge"
            )
        else:
            logging.warning(
                f"✅ Weight {weight_info.name} for input model has mutated during merge"
            )

        writer.save_tensor(weight_info.name, w)
    writer.finalize()

    tokenizer = AutoTokenizer.from_pretrained(secondary_model_path)
    tokenizer.save_pretrained(out_path, safe_serialization=True)

    model_out_config = secondary_model.config(
        trust_remote_code=merge_options.trust_remote_code
    )
    if dtype:
        model_out_config.torch_dtype = dtype
    model_out_config.save_pretrained(out_path)


main()
