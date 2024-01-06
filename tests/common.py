import os
import tempfile
from typing import Callable, Optional

from transformers import LlamaConfig, LlamaForCausalLM

from mergekit.config import MergeConfiguration
from mergekit.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex
from mergekit.merge import MergeOptions, run_merge


def run_and_check_merge(
    config: MergeConfiguration,
    check_nan: bool = True,
    validate: Optional[Callable[[str], None]] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=MergeOptions())
        assert os.path.exists(
            os.path.join(tmpdir, "model.safetensors.index.json")
        ), "No index file for merge"
        assert os.path.exists(
            os.path.join(tmpdir, "config.json")
        ), "No config json produced by merge"

        if check_nan:
            # check for NaN in output
            loader = LazyTensorLoader(
                ShardedTensorIndex.from_disk(tmpdir), lazy_unpickle=False
            )
            tp = loader.index.tensor_paths
            sorted_tensors = sorted(tp.keys(), key=lambda k: tp[k])
            for tensor_name in sorted_tensors:
                tensor = loader.get_tensor(tensor_name)
                has_nan = tensor.view(-1).isnan().any()
                assert not has_nan, "Output contains NaN"

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
