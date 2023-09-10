import copy
import json
import os.path
from typing import Dict, List, Optional
from typing_extensions import Annotated

import safetensors.torch
import torch
import transformers
import typer
import yaml
from pydantic import BaseModel
from tqdm import tqdm

from common import ModelReference
from sharded_tensor_index import LazyTensorLoader


class LayerSlice(BaseModel):
    model: str
    start: int
    end: int
    scale: Optional[float] = None


class BakllamaConfig(BaseModel):
    layer_slices: List[LayerSlice]
    embedding_source: Optional[str] = None
    lm_head_source: Optional[str] = None


class TensorWriter:
    out_path: str
    max_shard_size: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int

    def __init__(
        self, out_path: str, max_shard_size: int = 1000 * 1000 * 1000 * 5
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        tensor_size = tensor.view(-1).shape[0]
        if (
            self.current_shard
            and self.current_shard_size + tensor_size > self.max_shard_size
        ):
            self.flush_current_shard()

        if clone:
            tensor = tensor.clone()

        self.current_shard[name] = tensor
        self.current_shard_size += tensor_size

    def flush_current_shard(self):
        if not self.current_shard:
            return

        shard_name = f"model-{self.shards_written+1}.safetensors"
        for key in self.current_shard:
            self.weight_map[key] = shard_name
        safetensors.torch.save_file(
            self.current_shard,
            os.path.join(self.out_path, shard_name),
            metadata={"format": "pt"},
        )
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written = self.shards_written + 1

    def finalize(self):
        self.flush_current_shard()
        with open(
            os.path.join(self.out_path, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"metadata": {}, "weight_map": self.weight_map}, file)


def process(config: BakllamaConfig, out_path: str, clone_tensors: bool = False):
    if config.embedding_source is None:
        config.embedding_source = config.layer_slices[0].model

    if config.lm_head_source is None:
        config.lm_head_source = config.layer_slices[-1].model

    referenced_models = list(
        set(
            [s.model for s in config.layer_slices]
            + [config.embedding_source, config.lm_head_source]
        )
    )

    configs: List[transformers.LlamaConfig] = [
        transformers.LlamaConfig.from_pretrained(m) for m in referenced_models
    ]

    (hidden_size, imm_size) = configs[0].hidden_size, configs[0].intermediate_size
    assert all(
        (cfg.hidden_size, cfg.intermediate_size) == (hidden_size, imm_size)
        for cfg in configs
    ), "Source models must all have same feature dimensions"

    loaders: Dict[str, LazyTensorLoader] = {}
    for model in referenced_models:
        ref = ModelReference(path=model)
        loaders[model] = LazyTensorLoader(ref.tensor_index())

    new_config = copy.deepcopy(configs[0])
    new_config.num_hidden_layers = sum([s.end - s.start for s in config.layer_slices])

    os.makedirs(out_path, exist_ok=True)
    new_config.save_pretrained(out_path)

    layer_sources = []
    for s in config.layer_slices:
        for source_idx in range(s.start, s.end):
            layer_sources.append((s.model, source_idx, s.scale))

    writer = TensorWriter(out_path)

    writer.save_tensor(
        "model.embed_tokens.weight",
        loaders[config.embedding_source].get_tensor("model.embed_tokens.weight"),
        clone=clone_tensors,
    )
    for layer_idx, (model_name, source_layer_idx, scale) in enumerate(
        tqdm(layer_sources)
    ):
        for tensor_name in [
            "input_layernorm",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_proj",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]:
            source_key = f"model.layers.{source_layer_idx}.{tensor_name}.weight"

            weight = loaders[model_name].get_tensor(source_key)
            if "down_proj" in tensor_name and scale is not None:
                weight *= scale

            dst_key = f"model.layers.{layer_idx}.{tensor_name}.weight"
            writer.save_tensor(dst_key, weight, clone=clone_tensors)

    writer.save_tensor(
        "model.norm.weight",
        loaders[config.lm_head_source].get_tensor("model.norm.weight"),
        clone=clone_tensors,
    )
    writer.save_tensor(
        "lm_head.weight",
        loaders[config.lm_head_source].get_tensor("lm_head.weight"),
        clone=clone_tensors,
    )
    writer.finalize()


def main(
    config_path: str,
    out_path: str,
    clone_tensors: Annotated[
        bool,
        typer.Option(
            help="Clone tensors before saving, to allow multiple occurrences of the same layer"
        ),
    ] = False,
):
    with open(config_path, "r", encoding="utf-8") as file:
        config = BakllamaConfig(**yaml.safe_load(file))

    process(config, out_path, clone_tensors=clone_tensors)


if __name__ == "__main__":
    typer.run(main)
