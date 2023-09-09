import logging
import os
import os.path
from typing import List, Optional

import huggingface_hub
import numpy as np
import peft
import torch
import transformers
from pydantic import BaseModel

from sharded_tensor_index import ShardedTensorIndex


class ModelReference(BaseModel):
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

        return ModelReference(path=out_path)

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        assert self.lora_path is None

        path = self.path
        if not os.path.exists(path):
            has_safetensors = any(
                fn.lower().endswith(".safetensors")
                for fn in huggingface_hub.list_repo_files(path, repo_type="model")
            )
            patterns = ["tokenizer.model", "*.json"]
            if has_safetensors:
                patterns.append("*.safetensors")
            else:
                patterns.append("*.bin")

            path = huggingface_hub.snapshot_download(
                path, cache_dir=cache_dir, allow_patterns=patterns
            )

        return ShardedTensorIndex.from_disk(path)

    @classmethod
    def parse(cls, value: str) -> "ModelReference":
        """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""

        chunks = value.split("+")
        if len(chunks) == 1:
            return ModelReference(path=value)
        elif len(chunks) == 2:
            return ModelReference(path=chunks[0], lora_path=chunks[1])
        raise ValueError(f"Can't parse {value}")

    def __str__(self) -> str:
        if self.lora_path:
            return f"{self.path}+{self.lora_path}"
        return self.path

    class Config:
        frozen = True
        allow_mutation = False


def dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    elif name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    raise RuntimeError(f'Unimplemented dtype "{name}"')


def rectify_embed_sizes(param_name: str, tensors: List[torch.Tensor]):
    if "lm_head" in param_name or "embed_tokens" in param_name:
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        if take_common_submatrix(tensors):
            logging.warning(
                f"Using common submatrix of size {tensors[0].shape} for {param_name}"
            )


def take_common_submatrix(tensors: List[torch.Tensor]) -> bool:
    min_size = [None, None]
    for t in tensors:
        for idx in range(2):
            if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                min_size[idx] = t.shape[idx]

    if not all(t.shape == min_size for t in tensors):
        for idx in range(len(tensors)):
            tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]
        return True
    return False


def gradient_weights(gradient: List[float], num_samples: int) -> List[float]:
    assert len(gradient) > 1, "Need at least two values to define gradient"

    samples_per_weight = num_samples // (len(gradient) - 1)

    res = []
    for y0, y1 in zip(gradient[:-1], gradient[1:]):
        res.extend(np.linspace(y0, y1, num=samples_per_weight))
    while len(res) < num_samples:
        res.append(gradient[-1])
    return res


LLAMA_LAYER_MEMBERS: List[str] = [
    "input_layernorm",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
