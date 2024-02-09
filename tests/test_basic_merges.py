from typing import Dict, Optional

import pytest
from common import make_picollama, run_and_check_merge
from transformers import AutoConfig

from mergekit.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
    ParameterSetting,
)


@pytest.fixture(scope="session")
def model_a(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_a"))


@pytest.fixture(scope="session")
def model_b(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_b"))


@pytest.fixture(scope="session")
def model_c(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_c"))


class TestBasicMerges:
    def test_gpt2_copy(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model="gpt2")],
            dtype="bfloat16",
        )
        run_and_check_merge(config)

    def test_gpt2_stack(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[InputSliceDefinition(model="gpt2", layer_range=[0, 12])]
                )
            ]
            * 2,
            dtype="bfloat16",
        )

        def _check_config_layers(p: str):
            config = AutoConfig.from_pretrained(p)
            assert config.n_layer == 24

        run_and_check_merge(config, validate=_check_config_layers)

    def test_linear_merge(self, model_a, model_b):
        config = self.two_model_config(model_a, model_b, merge_method="linear")
        run_and_check_merge(config)

    def test_slerp_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="slerp", base_model=model_a
        )
        config.parameters = {"t": 0.35}
        run_and_check_merge(config)

    def test_task_arithmetic_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="task_arithmetic", base_model=model_c
        )
        run_and_check_merge(config)

    def test_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="ties",
            base_model=model_c,
            params={"density": 0.3},
        )
        run_and_check_merge(config)

    def test_dare_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="dare_ties",
            base_model=model_c,
            params={"density": 0.66},
        )
        run_and_check_merge(config)

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
