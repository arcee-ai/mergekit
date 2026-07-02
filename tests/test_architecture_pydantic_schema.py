from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
)


def test_configured_architecture_schema_rebuilds_cleanly():
    # transformers>=5 made PretrainedConfig a dataclass with a torch reference that's
    # only available under TYPE_CHECKING; pydantic's default schema resolution used to
    # choke on it (see arcee-ai/mergekit#681). A forced rebuild reproduces this
    # regardless of any caching from earlier, successful schema builds in-process.
    ConfiguredModuleArchitecture.model_rebuild(force=True)
    ConfiguredModelArchitecture.model_rebuild(force=True)
