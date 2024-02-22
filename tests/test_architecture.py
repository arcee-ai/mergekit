import pytest
from transformers import AutoConfig, LlamaConfig

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info


class TestArchitecture:
    def test_set_overrides(self):
        cfg = AutoConfig.from_pretrained("gpt2")
        arch_info = get_architecture_info(cfg)
        configured_arch_info = ConfiguredArchitectureInfo(info=arch_info, config=cfg)

        overrides = {"a_${layer_index}": "b_${layer_index}"}
        new_config = configured_arch_info.set_overrides(overrides)

        assert new_config.overrides == {f"a_{i}": f"b_{i}" for i in range(12)}
