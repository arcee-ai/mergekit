#!/usr/bin/env python3
# Charles O. Goddard
# 8/20/230
"""Merge a set of task-specific models into a base model using the methodology
of \"Resolving Interference When Merging Models\" (https://arxiv.org/abs/2306.01708)."""

import json
import logging
import os
import os.path
from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional

import huggingface_hub
import peft
import safetensors.torch
import torch
import transformers
import typer
from tqdm import tqdm

from sharded_tensor_index import LazyTensorLoader, ShardedTensorIndex


def main(
    base_model: Annotated[str, typer.Argument(help="Base model for merge")],
    out_path: Annotated[str, typer.Argument(help="Output directory for final model")],
    merge: Annotated[
        List[str], typer.Option(help="Add a model to the merge", metavar="MODEL")
    ],
    density: Annotated[
        List[float],
        typer.Option(
            help="Fraction of weights to keep for each model (default 0.33)",
            default_factory=list,
            show_default=False,
        ),
    ],
    cache_dir: Annotated[
        Optional[str], typer.Option(help="Override storage path for downloaded models")
    ] = None,
    merged_cache_dir: Annotated[
        Optional[str], typer.Option(help="Storage path for merged LoRA models")
    ] = None,
    cuda: bool = False,
    int8_mask: Annotated[
        bool, typer.Option(help="Store intermediate masks in int8 to save memory")
    ] = False,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16")] = True,
    naive_count: Annotated[
        bool, typer.Option(help="Use naive sign count instead of weight")
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy base model tokenizer into output")
    ] = True,
):
    """Merge a set of models with a shared base model by resolving sign differences."""
    if merged_cache_dir is None:
        merged_cache_dir = cache_dir

    if not density:
        density = [0.33] * len(merge)
    elif len(density) == 1:
        density = [density[0]] * len(merge)
    elif len(density) != len(merge):
        raise RuntimeError(
            "Must specify either one single density or exactly one per model"
        )

    logging.info(f"densities: {list(zip(merge, density))}")

    base_model: ModelReference = parse_model(base_model).merged(merged_cache_dir)
    base_index: ShardedTensorIndex = base_model.tensor_index(cache_dir)
    loaders = [
        LazyTensorLoader(
            parse_model(m).merged(merged_cache_dir).tensor_index(cache_dir)
        )
        for m in merge
    ]

    math_dev = "cuda" if cuda else "cpu"
    os.makedirs(out_path, exist_ok=True)

    weight_map = {}

    mask_dtype: Optional[torch.dtype] = torch.int8 if int8_mask else None

    unique_shards = list(sorted(set(base_index.tensor_paths.values())))
    for base_shard in unique_shards:
        logging.info(f"Processing shard {base_shard}")
        base_tensors = base_index.load_shard(base_shard)

        new_shard_name = shard_name_to_st(base_shard)
        tensors_out = {}

        for key in tqdm(base_tensors):
            logging.debug(f"- {key}")
            ty = key_dtype(key, bf16=bf16)

            b = base_tensors[key].to(math_dev).float()

            deltas = []
            for loader, model_name, model_density in zip(loaders, merge, density):
                try:
                    x = loader.get_tensor(key).to(math_dev).float()
                except Exception:
                    logging.warning(f"{model_name} has no tensor {key}")
                    continue

                if x.shape != b.shape:
                    if "lm_head" in key or "embed_tokens" in key:
                        x = x[: b.shape[0], : b.shape[1]]
                        logging.warning(f"Using submatrix of {model_name}:{key}")
                    else:
                        logging.warning(
                            f"skipping {model_name}:{key} due to size mismatch"
                        )
                        continue
                delta = sparsify(x - b, model_density)
                deltas.append(delta)

            if deltas:
                deltas = torch.stack(deltas, dim=0)
                mask = get_mask(
                    deltas,
                    method="count" if naive_count else "weight",
                    mask_dtype=mask_dtype,
                )
                new_deltas = (deltas * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                res = b + new_deltas
            else:
                res = b

            tensors_out[key] = res.to(ty).cpu()
            weight_map[key] = new_shard_name

        safetensors.torch.save_file(
            tensors_out,
            os.path.join(out_path, new_shard_name),
            metadata={"format": "pt"},
        )

    with open(
        os.path.join(out_path, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as fd:
        json.dump({"metadata": {}, "weight_map": weight_map}, fd)

    try:
        cfg = transformers.AutoConfig.from_pretrained(base_model.path)
        cfg.save_pretrained(out_path)
    except Exception as e:
        logging.warning("Failed to copy config from base model", exc_info=e)
        logging.warning(
            "The merge was still successful. "
            "Just copy config.json from one of the models, it's fine."
        )

    if copy_tokenizer:
        tok = transformers.AutoTokenizer.from_pretrained(base_model.path)
        tok.save_pretrained(out_path)

    logging.info("Merge complete")


@dataclass
class ModelReference:
    """A reference to a language model.

    Can be a hf hub path (username/repo), or local. Optionally includes a LoRA."""

    path: str
    lora_path: Optional[str] = None

    def merged(self, cache_dir: Optional[str] = None) -> "ModelReference":
        """Merge the LoRA if applicable and return a reference to the result."""
        if not self.lora_path:
            return self

        if not cache_dir:
            raise RuntimeError("Need to specify cache dir to merge adapters")

        out_path = os.path.join(
            cache_dir,
            os.path.basename(self.path) + "_" + os.path.basename(self.lora_path),
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            logging.info(f"Loading {self.path} for merge...")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            model = peft.PeftModel.from_pretrained(
                model, self.lora_path, is_trainable=False
            )
            logging.info(f"Merging {self.lora_path} into {self.path}")
            model = model.merge_and_unload()
            model.save_pretrained(out_path, safe_serialization=True)
            del model

        return ModelReference(out_path)

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        assert self.lora_path is None

        path = self.path
        if not os.path.exists(path):
            path = huggingface_hub.snapshot_download(path, cache_dir=cache_dir)

        return ShardedTensorIndex.from_disk(path)


def parse_model(value: str):
    """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""

    chunks = value.split("+")
    if len(chunks) == 1:
        return ModelReference(value)
    elif len(chunks) == 2:
        return ModelReference(chunks[0], lora_path=chunks[1])
    else:
        raise ValueError(f"Can't parse {value}")


def key_dtype(key: str, bf16: bool = True) -> torch.dtype:
    """Determine what precision to store a tensor with the given key in.

    Fairly specialized to Llama models."""
    if key.endswith(".invfreq") or "embed_tokens" in key or "lm_head" in key:
        return torch.float32
    return torch.bfloat16 if bf16 else torch.float16


def shard_name_to_st(name: str) -> str:
    if name.endswith(".bin"):
        name = name[: -len(".bin")] + ".safetensors"
    return name.replace("pytorch_model", "model")


def sparsify(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def get_mask(
    delta: torch.Tensor,
    method: Literal["weight", "count"] = "weight",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the paper use 'weight'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "weight":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
