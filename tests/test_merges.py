import os
import tempfile
from typing import Dict, Optional

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from mergekit.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
    ParameterSetting,
)
from mergekit.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex
from mergekit.merge import MergeOptions, run_merge


def make_picollama(path: str):
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=16,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def model_a(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_a"))


@pytest.fixture(scope="session")
def model_b(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_b"))


@pytest.fixture(scope="session")
def model_c(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_c"))


class TestMerges:
    def test_gpt2_copy(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model="gpt2")],
            dtype="bfloat16",
        )
        self.run_and_check_merge(config)

    def test_gpt2_stack(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[InputSliceDefinition(model="gpt2", layer_range=[0, 12])]
                    * 2
                )
            ],
            dtype="bfloat16",
        )
        self.run_and_check_merge(config)

    def test_linear_merge(self, model_a, model_b):
        config = self.two_model_config(model_a, model_b, merge_method="linear")
        self.run_and_check_merge(config)

    def test_slerp_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="slerp", base_model=model_a
        )
        config.parameters = {"t": 0.35}
        self.run_and_check_merge(config)

    def test_task_arithmetic_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="task_arithmetic", base_model=model_c
        )
        self.run_and_check_merge(config)

    def test_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="ties",
            base_model=model_c,
            params={"density": 0.3},
        )
        self.run_and_check_merge(config)

    def test_dare_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="dare_ties",
            base_model=model_c,
            params={"density": 0.66},
        )
        self.run_and_check_merge(config)

    def run_and_check_merge(self, config: MergeConfiguration):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_merge(config, out_path=tmpdir, options=MergeOptions())
            assert os.path.exists(
                os.path.join(tmpdir, "model.safetensors.index.json")
            ), "No index file for merge"
            assert os.path.exists(
                os.path.join(tmpdir, "config.json")
            ), "No config json produced by merge"

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

    def two_model_config(
        self,
        model_a,
        model_b,
        merge_method: str,
        base_model: Optional[str] = None,
        params: Optional[Dict[str, ParameterSetting]] = None,
    ):
        config = MergeConfiguration(
            merge_method=merge_method,
            base_model=base_model,
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={"weight": 0.6},
                ),
                InputModelDefinition(
                    model=model_b,
                    parameters={"weight": 0.4},
                ),
            ],
            dtype="bfloat16",
            parameters=params,
        )

        return config
