import logging
import os
import os.path
from typing import Dict, List, Optional

import huggingface_hub
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

        return ModelReference(out_path)

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
            print(patterns)

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
        else:
            raise ValueError(f"Can't parse {value}")

    def __str__(self) -> str:
        if self.lora_path:
            return f"{self.path}+{self.lora_path}"
        return self.path

    class Config:
        allow_mutation = False
        frozen = True


def dtype_from_name(s: Optional[str]) -> torch.dtype:
    if s == "bfloat16":
        return torch.bfloat16
    elif s == "float16":
        return torch.float16
    elif s == "float32":
        return torch.float32
    else:
        raise RuntimeError(f'Unimplemented dtype "{s}"')
