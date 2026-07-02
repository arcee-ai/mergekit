import pytest
from pydantic import ValidationError

from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
    ModuleArchitecture,
)


def test_configured_architecture_schema_rebuilds_cleanly():
    # transformers>=5 made PretrainedConfig a dataclass with a torch reference only available under TYPE_CHECKING, which used to break pydantic's schema resolution (arcee-ai/mergekit#681); a forced rebuild reproduces this regardless of caching from an earlier successful build.
    ConfiguredModuleArchitecture.model_rebuild(force=True)
    ConfiguredModelArchitecture.model_rebuild(force=True)


class _StubModuleArchitecture(ModuleArchitecture):
    def pre_weights(self, config):
        return []

    def post_weights(self, config):
        return []

    def layer_weights(self, index, config):
        return []


def test_configured_module_architecture_rejects_non_pretrainedconfig():
    with pytest.raises(ValidationError):
        ConfiguredModuleArchitecture(
            info=_StubModuleArchitecture(), config="not a config"
        )
