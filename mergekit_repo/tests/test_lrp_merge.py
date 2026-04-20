# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

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
    make_picollama,
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


class TestLRPMerge:
    def test_lrp_merge_basic(self, model_a, model_b, model_c):
        """Test basic LRP merge with two models and a base model."""
        config = self.two_model_config(
            model_a, model_b, merge_method="lrp", base_model=model_c
        )
        run_and_check_merge(config)

    def test_lrp_merge_density_param(self, model_a, model_b, model_c):
        """Test LRP merge with custom density parameter."""
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="lrp",
            base_model=model_c,
            params={"density": 0.5},
        )
        run_and_check_merge(config)

    def test_lrp_merge_high_density(self, model_a, model_b, model_c):
        """Test LRP merge with high density (keep 90% of weights)."""
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="lrp",
            base_model=model_c,
            params={"density": 0.9},
        )
        run_and_check_merge(config)

    def test_lrp_merge_low_density(self, model_a, model_b, model_c):
        """Test LRP merge with low density (keep 30% of weights)."""
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="lrp",
            base_model=model_c,
            params={"density": 0.3},
        )
        run_and_check_merge(config)

    def test_lrp_merge_invalid_density_too_high(self, model_a, model_b, model_c):
        """Test that density > 1.0 raises an error."""
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="lrp",
            base_model=model_c,
            params={"density": 1.5},
        )
        with pytest.raises(ValueError):
            run_and_check_merge(config)

    def test_lrp_merge_invalid_density_negative(self, model_a, model_b, model_c):
        """Test that negative density raises an error."""
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="lrp",
            base_model=model_c,
            params={"density": -0.1},
        )
        with pytest.raises(ValueError):
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
