from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from mergekit import merge_methods
from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.merge_methods import MergeBatch, TensorBatch, merge_state_dicts
from mergekit.options import MergeOptions
from mergekit.scripts.merge_raw_pytorch import RawPyTorchMergeConfig, plan_flat_merge
from mergekit.tokenizer import PermutedEmbeddings
from tests.common import make_picollama, run_and_check_merge


def mixed_models():
    return [
        {
            "low": torch.tensor([2.0], dtype=torch.bfloat16),
            "mixed": torch.tensor([2.0], dtype=torch.bfloat16),
            "norm": torch.tensor([2.0], dtype=torch.float32),
            "double": torch.tensor([2.0], dtype=torch.float64),
            "counter": torch.tensor(7),
        },
        {
            "low": torch.tensor([4.0], dtype=torch.bfloat16),
            "mixed": torch.tensor([4.0], dtype=torch.float16),
            "norm": torch.tensor([4.0], dtype=torch.float32),
            "double": torch.tensor([4.0], dtype=torch.float32),
            "counter": torch.tensor(7),
        },
    ]


@pytest.mark.parametrize("method", ["linear", "task_arithmetic"])
def test_state_dict_promotes_each_weight_independently(method):
    models = mixed_models()
    result = merge_state_dicts(models, method, base=0, parameters={"weight": 0.5})
    for name, dtype in (
        ("low", torch.bfloat16),
        ("mixed", torch.float32),
        ("norm", torch.float32),
        ("double", torch.float64),
    ):
        torch.testing.assert_close(result[name], torch.tensor([3.0], dtype=dtype))
    assert result["counter"].dtype == torch.int64
    assert models[0]["mixed"].dtype == torch.bfloat16


def test_input_cast_and_output_cast_have_distinct_numerics():
    models = [{"w": torch.tensor([1.003]), "counter": torch.tensor(7)}]
    parameters = {"scale": 1000.0}
    input_cast = merge_state_dicts(
        models, "passthrough", parameters=parameters, dtype=torch.bfloat16
    )
    output_cast = merge_state_dicts(
        models, "passthrough", parameters=parameters, out_dtype=torch.bfloat16
    )
    assert input_cast["w"].item() == 1000.0
    assert output_cast["w"].item() == 1004.0
    assert input_cast["counter"].item() == output_cast["counter"].item() == 7
    assert input_cast["counter"].dtype == output_cast["counter"].dtype == torch.int64
    both = merge_state_dicts(
        models,
        "passthrough",
        parameters=parameters,
        dtype=torch.bfloat16,
        out_dtype=torch.float64,
    )
    assert both["w"].dtype == torch.float64
    assert both["w"].item() == 1000.0


@pytest.mark.parametrize("option", ["dtype", "out_dtype"])
def test_state_dict_rejects_nonfloating_dtype_overrides(option):
    with pytest.raises(ValueError, match="floating-point torch.dtype"):
        merge_state_dicts(mixed_models(), "linear", **{option: torch.int64})


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
@pytest.mark.parametrize("out_dtype", [None, "float64"])
def test_raw_graph_dtype_policy(tmp_path, dtype, out_dtype):
    models = mixed_models()
    paths = []
    for index, tensors in enumerate(models):
        path = tmp_path / f"model{index}.safetensors"
        save_file({k: v for k, v in tensors.items() if k != "counter"}, path)
        paths.append(str(path))
    config = RawPyTorchMergeConfig(
        merge_method="linear",
        models=[{"model": p} for p in paths],
        parameters={"weight": 1.0},
        dtype=dtype,
        out_dtype=out_dtype,
    )
    output = tmp_path / "output"
    tasks = plan_flat_merge(config, str(output), False, False, MergeOptions())
    Executor(tasks, math_device="cpu", storage_device="cpu").execute()
    actual = load_file(output / "model.safetensors")
    expected = merge_state_dicts(
        models,
        "linear",
        parameters={"weight": 1.0},
        dtype=getattr(torch, dtype) if dtype else None,
        out_dtype=getattr(torch, out_dtype) if out_dtype else None,
    )
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_linear_ignores_ambient_autocast(device, autocast_dtype):
    tensors = [torch.tensor([100000.0, 1.001, 1.002], device=device)] * 2
    method = merge_methods.get("linear")
    with torch.autocast(device, dtype=autocast_dtype):
        actual = method(MergeBatch.from_tensors(tensors), weight=[0.5, 0.5]).one()
    torch.testing.assert_close(actual, tensors[0], rtol=0, atol=0)


@torch.inference_mode()
def test_linear_low_precision_has_no_full_float32_scratch():
    source = torch.ones(1, 2, 1024, 1024, dtype=torch.bfloat16)
    coefficients = torch.tensor([[0.5, 0.5]])
    with torch.profiler.profile(profile_memory=True) as profile:
        actual = merge_methods.get("linear").merge_batch(
            TensorBatch(source), weight=coefficients
        )
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, source[:, 0])
    # A full float32 conversion of either input is larger than the output.
    assert max(event.cpu_memory_usage for event in profile.events()) <= actual.nbytes


def test_embedding_alignment_promotes_before_copying():
    refs = [ModelReference.model_validate(name) for name in ("a", "b")]
    inputs = {
        refs[0]: torch.tensor([[1.0]], dtype=torch.bfloat16),
        refs[1]: torch.tensor([[1.001]], dtype=torch.float32),
    }
    task = PermutedEmbeddings.model_construct(
        tokens=None, pad_to_multiple_of=None, base_model=None
    )
    info = SimpleNamespace(
        tokenizer=SimpleNamespace(get_vocab=lambda: {"token": 0}),
        permutations={ref: {0: 0} for ref in refs},
    )
    actual = task.execute(info, inputs)
    for ref in refs:
        torch.testing.assert_close(actual[ref], inputs[ref].float())


@pytest.mark.parametrize("dtype,out_dtype", [(None, None), ("bfloat16", "float32")])
def test_yaml_mixed_checkpoint_precisions(tmp_path, dtype, out_dtype):
    paths = [make_picollama(tmp_path / f"model{i}") for i in range(2)]
    path = tmp_path / "model0" / "model.safetensors"
    tensors = load_file(path)
    # Preserve float32 norms within the otherwise bfloat16 checkpoint.
    save_file(
        {
            name: tensor if "norm" in name else tensor.bfloat16()
            for name, tensor in tensors.items()
        },
        path,
    )
    config = MergeConfiguration(
        merge_method="linear",
        models=[{"model": path} for path in paths],
        parameters={"weight": 1.0},
        dtype=dtype,
        out_dtype=out_dtype,
    )

    def check(output):
        from mergekit.io import LazyTensorLoader

        loader = LazyTensorLoader.from_disk(output)
        for name in loader.index.tensor_paths:
            assert loader.get_tensor(name).dtype == torch.float32

    run_and_check_merge(config, validate=check)
