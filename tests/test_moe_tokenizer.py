import pytest

import mergekit.scripts.moe as moe_script
from mergekit.moe.config import Expert, MoEMergeConfig
from mergekit.options import MergeOptions


class _StopBuild(Exception):
    pass


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_moe_tokenizer_follows_trust_remote_code(monkeypatch, trust_remote_code):
    """The base model's tokenizer is loaded with the --trust-remote-code
    setting, like every other tokenizer load in mergekit, so a base model with
    a custom tokenizer can be used once the user has opted in."""
    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(
        moe_script.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(moe_script, "select_output_arch", lambda *a, **k: None)

    config = MoEMergeConfig(
        base_model="base/model",
        experts=[
            Expert(source_model="a/model", positive_prompts=["a"]),
            Expert(source_model="b/model", positive_prompts=["b"]),
        ],
        gate_mode="random",
    )
    with pytest.raises(_StopBuild):
        moe_script.build(
            config,
            "/tmp/out",
            MergeOptions(trust_remote_code=trust_remote_code),
        )
    assert seen.get("trust_remote_code", False) is trust_remote_code
