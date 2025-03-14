import os
import tempfile
from typing import Callable, Optional

from transformers import (
    AutoConfig,
    CLIPVisionConfig,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

from mergekit.architecture import (
    arch_info_for_config,
    get_architecture_info,
)
from mergekit.config import MergeConfiguration
from mergekit.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex
from mergekit.merge import MergeOptions, run_merge


def run_and_check_merge(
    config: MergeConfiguration,
    check_nan: bool = True,
    check_tensors: bool = True,
    validate: Optional[Callable[[str], None]] = None,
    index_json_name: Optional[str] = None,
    auto_arch: bool = False,
):
    if index_json_name is None:
        index_json_name = "model.safetensors.index.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=MergeOptions())
        index_path = os.path.join(tmpdir, index_json_name)
        index_exists = os.path.exists(index_path)
        single_shard_exists = os.path.exists(index_path.replace(".index.json", ""))
        assert index_exists or single_shard_exists, "No model produced by merge"
        assert os.path.exists(
            os.path.join(tmpdir, "config.json")
        ), "No config json produced by merge"

        if check_nan:
            # check for NaN in output
            loader = LazyTensorLoader.from_disk(tmpdir, lazy_unpickle=False)
            tp = loader.index.tensor_paths
            sorted_tensors = sorted(tp.keys(), key=lambda k: tp[k])
            for tensor_name in sorted_tensors:
                tensor = loader.get_tensor(tensor_name)
                has_nan = tensor.view(-1).isnan().any()
                assert not has_nan, "Output contains NaN"

        if check_tensors:
            model_config = AutoConfig.from_pretrained(tmpdir)
            if auto_arch:
                arch_info = get_architecture_info(config, MergeOptions())
            else:
                arch_info = arch_info_for_config(model_config)

            index = ShardedTensorIndex.from_disk(tmpdir)
            for weight_info in arch_info.all_weights(model_config):
                if weight_info.optional:
                    continue
                if weight_info.name not in index.tensor_paths and not any(
                    a in index.tensor_paths for a in weight_info.aliases
                ):
                    raise RuntimeError(f"Output missing tensor {weight_info.name}")

        if validate:
            validate(tmpdir)


def make_picollama(path: str, vocab_size: int = 64):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=16,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def make_gpt2size(path: str):
    cfg = GPT2Config(
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        n_positions=1024,
        vocab_size=50257,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def make_picoLlaVa(path: str):
    # Define minimal vision configuration
    vision_config = CLIPVisionConfig(
        image_size=32,
        patch_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
    )

    # Define minimal text configuration
    text_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=16,
        num_hidden_layers=2,
    )

    # Combine into Llava configuration
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_seq_length=16,
    )

    # Instantiate the model
    model = LlavaForConditionalGeneration(config=llava_config)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)
