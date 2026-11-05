from unittest import mock

import pytest
import yaml
from click.testing import CliRunner
from transformers import LlamaConfig

from mergekit.common import ModelReference
from mergekit.scripts.layershuffle import main

NUM_LAYERS = 4


def _cfg() -> LlamaConfig:
    return LlamaConfig(num_hidden_layers=NUM_LAYERS, architectures=["LlamaForCausalLM"])


@pytest.mark.parametrize("full_random", [False, True])
def test_layershuffle_generates_config(full_random, tmp_path):
    args = ["-m", "org/a", "-m", "org/b", "-w", "0.5", "-w", "0.5", "--dry-run"]
    args.append("--print-yaml")
    if full_random:
        args.append("--full-random")

    with mock.patch.object(ModelReference, "config", return_value=_cfg()):
        result = CliRunner().invoke(main, [str(tmp_path / "out"), *args])

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.output)
    total = sum(
        s["sources"][0]["layer_range"][1] - s["sources"][0]["layer_range"][0]
        for s in data["slices"]
    )
    if full_random:
        assert total >= 1
    else:
        assert total == NUM_LAYERS


@pytest.mark.parametrize("flags", [[], ["--trust-remote-code"]])
@pytest.mark.parametrize("full_random", [False, True])
def test_layershuffle_forwards_trust_remote_code(flags, full_random, tmp_path):
    trust = bool(flags)
    args = ["-m", "org/a", "-m", "org/b", "-w", "0.5", "-w", "0.5", "--dry-run"]
    if full_random:
        args.append("--full-random")

    with mock.patch.object(ModelReference, "config", return_value=_cfg()) as m:
        result = CliRunner().invoke(main, [str(tmp_path / "out"), *args, *flags])

    assert result.exit_code == 0, result.output
    assert m.call_count >= 1
    for call in m.call_args_list:
        assert call.kwargs.get("trust_remote_code", False) is trust
