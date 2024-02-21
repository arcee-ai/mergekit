import pytest
from transformers import LlamaConfig

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info


class TestArchitecture:
    def test_set_overrides(self):
        cfg = LlamaConfig(vocab_size=64, hidden_size=32)
        arch_info = get_architecture_info(cfg)
        configured_arch_info = ConfiguredArchitectureInfo(arch_info, cfg)

        overrides = {"a_{layer_index}": "b_layer_{layer_index}"}
        new_config = configured_arch_info.set_overrides(overrides)

        assert new_config.overrides == overrides
