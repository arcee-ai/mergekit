from typing import Dict, Optional

import pytest
from transformers import AutoConfig

from mergekit.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
    ParameterSetting,
)
from mergekit.io import LazyTensorLoader
from tests.common import (
    make_gpt2size,
    make_picollama,
    make_picoLlaVa,
    run_and_check_merge,
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


@pytest.fixture(scope="session")
def vlm_a(tmp_path_factory):
    return make_picoLlaVa(tmp_path_factory.mktemp("vlm_a"))


@pytest.fixture(scope="session")
def vlm_b(tmp_path_factory):
    return make_picoLlaVa(tmp_path_factory.mktemp("vlm_b"))


@pytest.fixture(scope="session")
def vlm_c(tmp_path_factory):
    return make_picoLlaVa(tmp_path_factory.mktemp("vlm_c"))


@pytest.fixture(scope="session")
def gpt2_like(tmp_path_factory):
    return make_gpt2size(tmp_path_factory.mktemp("gpt2_like"))


class TestBasicMerges:
    def test_gpt2_copy(self, gpt2_like):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=gpt2_like)],
            dtype="bfloat16",
        )
        run_and_check_merge(config)

    def test_gpt2_stack(self, gpt2_like):
        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[InputSliceDefinition(model=gpt2_like, layer_range=[0, 12])]
                )
            ]
            * 2,
            dtype="bfloat16",
        )

        def _check_config_layers(p: str):
            config = AutoConfig.from_pretrained(p)
            assert config.n_layer == 24

        run_and_check_merge(config, validate=_check_config_layers)

    def test_passthrough_scale(self, model_a):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={
                        "scale": [
                            {"filter": "o_proj", "value": 0},
                            {"value": 1},
                        ]
                    },
                )
            ],
        )

        def _check_o_proj(p: str):
            loader = LazyTensorLoader.from_disk(p)
            saw_any = False
            for name in loader.index.tensor_paths:
                if "o_proj" in name:
                    param = loader.get_tensor(name)
                    assert (param == 0).all()
                    saw_any = True
                elif "lm_head" in name:
                    param = loader.get_tensor(name)
                    assert param.count_nonzero() > 0

            assert saw_any, "No o_proj parameters found"

        run_and_check_merge(config, validate=_check_o_proj)

    def test_linear_merge(self, model_a, model_b):
        config = self.two_model_config(model_a, model_b, merge_method="linear")
        run_and_check_merge(config)

    def test_slerp_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="slerp", base_model=model_a
        )
        config.parameters = {"t": 0.35}
        run_and_check_merge(config)

    def test_nuslerp_merges(self, model_a, model_b, model_c):
        for base_model in [None, model_c]:
            for row_wise in [False, True]:
                for flatten in [False, True]:
                    print(
                        f"Testing nuslerp with row_wise={row_wise}, flatten={flatten}, base_model={base_model}"
                    )
                    run_and_check_merge(
                        self.two_model_config(
                            model_a,
                            model_b,
                            merge_method="nuslerp",
                            base_model=base_model,
                            params={
                                "nuslerp_row_wise": row_wise,
                                "nuslerp_flatten": flatten,
                            },
                        )
                    )

        # test weights that sum to zero
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="nuslerp",
            base_model=model_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        config.models[0].parameters["weight"] = -0.5
        config.models[1].parameters["weight"] = 0.5
        run_and_check_merge(config)

    def test_task_arithmetic_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="task_arithmetic", base_model=model_c
        )
        run_and_check_merge(config)

    def test_breadcrumbs_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="breadcrumbs", base_model=model_c
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

    def test_sce_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="sce",
            base_model=model_c,
            params={"select_topk": 0.5},
        )
        run_and_check_merge(config)

    def test_multislerp_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="multislerp",
            base_model=model_c,
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

    def test_model_stock_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_b, model_c, merge_method="model_stock", base_model=model_a
        )
        run_and_check_merge(config)

    def test_model_stock_filterwise_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_b,
            model_c,
            merge_method="model_stock",
            base_model=model_a,
            params={"filter_wise": True},
        )
        run_and_check_merge(config)

    def test_arcee_fusion_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="arcee_fusion", base_model=model_a
        )
        run_and_check_merge(config)

    def test_nearswap_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="nearswap",
            base_model=model_a,
            params={"t": 0.0001},
        )
        run_and_check_merge(config)

    def test_della_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="della",
            base_model=model_c,
            params={"density": 0.66, "epsilon": 0.05, "lambda": 0.5},
        )
        run_and_check_merge(config)

    def test_karcher_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="karcher",
            base_model=model_c,
            params={"max_iter": 5, "tol": 1e-5},
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

    def test_linear_VLM_merge(self, vlm_a, vlm_b):
        config = self.two_model_config(vlm_a, vlm_b, merge_method="linear")
        run_and_check_merge(config, auto_arch=True)

    def test_slerp_VLM_merge(self, vlm_a, vlm_b):
        config = self.two_model_config(
            vlm_a, vlm_b, merge_method="slerp", base_model=vlm_a
        )
        config.parameters = {"t": 0.35}
        run_and_check_merge(config, auto_arch=True)

    def test_nuslerp_VLM_merges(self, vlm_a, vlm_b, vlm_c):
        for base_model in [None, vlm_c]:
            for row_wise in [False, True]:
                for flatten in [False, True]:
                    print(
                        f"Testing nuslerp with row_wise={row_wise}, flatten={flatten}, base_model={base_model}"
                    )
                    run_and_check_merge(
                        self.two_model_config(
                            vlm_a,
                            vlm_b,
                            merge_method="nuslerp",
                            base_model=base_model,
                            params={
                                "nuslerp_row_wise": row_wise,
                                "nuslerp_flatten": flatten,
                            },
                        ),
                        auto_arch=True,
                    )

        # test weights that sum to zero
        config = self.two_model_config(
            vlm_a,
            vlm_b,
            merge_method="nuslerp",
            base_model=vlm_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        config.models[0].parameters["weight"] = -0.5
        config.models[1].parameters["weight"] = 0.5
        run_and_check_merge(config, auto_arch=True)

    def test_task_arithmetic_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_a, vlm_b, merge_method="task_arithmetic", base_model=vlm_c
        )
        run_and_check_merge(config, auto_arch=True)

    def test_breadcrumbs_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_a, vlm_b, merge_method="breadcrumbs", base_model=vlm_c
        )
        run_and_check_merge(config, auto_arch=True)

    def test_ties_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_a,
            vlm_b,
            merge_method="ties",
            base_model=vlm_c,
            params={"density": 0.3},
        )
        run_and_check_merge(config, auto_arch=True)

    def test_dare_ties_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_a,
            vlm_b,
            merge_method="dare_ties",
            base_model=vlm_c,
            params={"density": 0.66},
        )
        run_and_check_merge(config, auto_arch=True)

    def test_model_stock_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_b, vlm_c, merge_method="model_stock", base_model=vlm_a
        )
        run_and_check_merge(config, auto_arch=True)

    def test_model_stock_filterwise_VLM_merge(self, vlm_a, vlm_b, vlm_c):
        config = self.two_model_config(
            vlm_b,
            vlm_c,
            merge_method="model_stock",
            base_model=vlm_a,
            params={"filter_wise": True},
        )
        run_and_check_merge(config, auto_arch=True)
