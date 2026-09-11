from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from mergekit import merge_methods
from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.merge_methods import MergeBatch, TensorBatch, merge_state_dicts
from mergekit.merge_methods.task_adapter import TensorDictWrapper
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
        merge_state_dicts(
            mixed_models(),
            "linear",
            parameters={"weight": 1.0},
            **{option: torch.int64},
        )


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
@pytest.mark.parametrize("out_dtype", [None, "float64"])
@pytest.mark.parametrize("method_name", ["linear", "task_arithmetic"])
def test_raw_graph_dtype_policy(tmp_path, monkeypatch, dtype, out_dtype, method_name):
    models = mixed_models()
    paths = []
    for index, tensors in enumerate(models):
        path = tmp_path / f"model{index}.safetensors"
        save_file(tensors, path)
        paths.append(str(path))

    # Inspect gathered inputs before Executor can transfer them to the math device.
    # This catches late downcasts without requiring a GPU or allocation thresholds.
    gather = TensorDictWrapper.execute
    seen = set()

    def checked_gather(self, **kwargs):
        tensors = gather(self, **kwargs)
        for model, tensor in tensors.items():
            name = self.tensors[model].tensor_name
            source = models[paths.index(model.model.path)][name]
            expected = (
                source.to(getattr(torch, dtype))
                if dtype and source.is_floating_point()
                else source
            )
            assert tensor.device.type == "cpu"
            torch.testing.assert_close(tensor, expected)
            seen.add((model.model.path, name))
        return tensors

    monkeypatch.setattr(TensorDictWrapper, "execute", checked_gather)
    config = RawPyTorchMergeConfig(
        merge_method=method_name,
        models=[{"model": p} for p in paths[1:]],
        base_model=paths[0],  # The implicitly added base must also be cast early.
        parameters={"weight": 1.0},
        dtype=dtype,
        out_dtype=out_dtype,
    )
    output = tmp_path / "output"
    tasks = plan_flat_merge(config, str(output), False, False, MergeOptions())
    Executor(tasks, math_device="cpu", storage_device="cpu").execute()
    assert seen == {
        (path, name) for path, model in zip(paths, models) for name in model
    }
    actual = load_file(output / "model.safetensors")
    expected = merge_state_dicts(
        models,
        method_name,
        base=0,
        parameters={"weight": 1.0},
        dtype=getattr(torch, dtype) if dtype else None,
        out_dtype=getattr(torch, out_dtype) if out_dtype else None,
    )
    torch.testing.assert_close(actual, expected)


def test_raw_graph_rejects_nonfloating_input_dtype(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"w": torch.tensor([1.5])}, path)
    config = RawPyTorchMergeConfig(
        merge_method="passthrough",
        models=[{"model": str(path)}],
        dtype="int64",
    )
    tasks = plan_flat_merge(
        config, str(tmp_path / "output"), False, False, MergeOptions()
    )
    with pytest.raises(ValueError, match="floating-point torch.dtype"):
        Executor(tasks).execute()


@pytest.mark.parametrize(
    "method_name", ["linear", "slerp", "task_arithmetic", "passthrough"]
)
def test_raw_graph_merges_batchnorm_without_casting_buffers(tmp_path, method_name):
    models = [
        torch.nn.BatchNorm1d(2) for _ in range(1 if method_name == "passthrough" else 2)
    ]
    paths = []
    for index, model in enumerate(models):
        model.register_buffer("enabled", torch.tensor([True, False]))
        model.num_batches_tracked.fill_(257)
        model.running_mean.fill_(index + 1.003)
        path = tmp_path / f"model{index}.safetensors"
        save_file(model.state_dict(), path)
        paths.append(str(path))
    parameters = (
        {"scale": 2.0}
        if method_name == "passthrough"
        else {"t": 0.3} if method_name == "slerp" else {"weight": 0.25}
    )
    config = RawPyTorchMergeConfig(
        merge_method=method_name,
        models=[{"model": path} for path in paths],
        base_model=paths[0] if len(models) > 1 else None,
        parameters=parameters,
        dtype="bfloat16",
        out_dtype="float64",
    )
    output = tmp_path / "output"
    Executor(
        plan_flat_merge(config, str(output), False, False, MergeOptions())
    ).execute()
    actual = load_file(output / "model.safetensors")
    expected = merge_state_dicts(
        models,
        method_name,
        base=0 if len(models) > 1 else None,
        parameters=parameters,
        dtype=torch.bfloat16,
        out_dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected)
    assert actual["num_batches_tracked"].dtype == torch.int64
    assert actual["num_batches_tracked"].item() == 257
    assert actual["enabled"].dtype == torch.bool


@pytest.mark.parametrize(
    "other",
    [
        torch.tensor(2**24),  # A float32 input cast would hide this difference.
        torch.tensor(2**24 + 1, dtype=torch.int32),
        torch.tensor([2**24 + 1]),
        torch.tensor(2**24 + 1, dtype=torch.float64),
    ],
    ids=["value", "dtype", "shape", "mixed_float_and_integer"],
)
def test_raw_graph_rejects_differing_buffers_before_casting(tmp_path, other):
    paths = []
    for index, counter in enumerate([torch.tensor(2**24 + 1), other]):
        path = tmp_path / f"model{index}.safetensors"
        save_file({"counter": counter}, path)
        paths.append(str(path))
    config = RawPyTorchMergeConfig(
        merge_method="linear",
        models=[{"model": path} for path in paths],
        parameters={"weight": 1.0},
        dtype="float32",
        out_dtype="bfloat16",
    )
    tasks = plan_flat_merge(
        config, str(tmp_path / "output"), False, False, MergeOptions()
    )
    with pytest.raises(ValueError, match="Non-floating buffer 'counter' differs"):
        Executor(tasks).execute()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def device(request):
    return request.param


@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_linear_ignores_ambient_autocast(device, autocast_dtype):
    tensors = [torch.tensor([100000.0, 1.001, 1.002], device=device)] * 2
    method = merge_methods.get("linear")
    with torch.autocast(device, dtype=autocast_dtype):
        (actual,) = method(
            MergeBatch.from_tensors(tensors), parameters={"weight": [0.5, 0.5]}
        )
    torch.testing.assert_close(actual, tensors[0], rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("count", [2, 8])
def test_linear_scratch_is_bounded_independently_of_input_count(count):
    source = torch.ones(1, count, 1024, 1024, dtype=torch.bfloat16)
    coefficients = torch.full((1, count), 1.0 / count)
    with torch.profiler.profile(profile_memory=True) as profile:
        actual = merge_methods.get("linear").merge_batch(
            TensorBatch(tuple(source.unbind(1))), weight=coefficients
        )
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, source[:, 0])
    # CPU TensorIterator may cast the current input to FP64, alongside the FP64
    # accumulator. Scratch must stay bounded independently of the input count.
    live = peak = 0
    for event in profile.profiler.kineto_results.events():
        if event.name() == "[memory]":
            live += event.nbytes()
            peak = max(peak, live)
    assert peak <= 8 * actual.nbytes + 4096


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


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("direct", [False, True])
@torch.no_grad()
def test_dtype_intermediates_released_between_chunks(batched, dtype, device, direct):
    import weakref

    from mergekit.merge_methods import BatchOptions, TensorGroup, merge_method

    models = [
        {
            f"w{i}": torch.full((16,), i + offset, dtype=source_dtype, device=device)
            for i in range(5)
        }
        for offset, source_dtype in ((1, torch.float16), (3, torch.bfloat16))
    ]
    target_dtype = dtype or torch.float32
    previous = []
    calls = []

    def check_inputs(tensors):
        # Check owners, including the base of packed views, so a view cannot hide
        # retained storage. Sources and final (float16) outputs remain live.
        assert all(ref() is None for ref in previous)
        previous.clear()
        for tensor in tensors:
            assert tensor.dtype == target_dtype
            owner = tensor if tensor._base is None else tensor._base
            previous.append(weakref.ref(owner))
        calls.append(1)

    def batch_kernel(batch: TensorBatch) -> torch.Tensor:
        check_inputs(batch.tensors)
        assert batch.tensors[0].shape[0] == 1
        result = sum(batch.tensors)
        previous.append(weakref.ref(result))
        return result

    def group_kernel(group: TensorGroup) -> torch.Tensor:
        check_inputs(group.tensors)
        result = group.tensors[0] + group.tensors[1]
        previous.append(weakref.ref(result))
        return result

    kernel = batch_kernel if batched else group_kernel
    method = merge_method(kernel, name="check_lifetimes")
    options = dict(
        dtype=dtype,
        out_dtype=torch.float16,
        # Exactly one group in the target dtype fits. Using source byte sizes
        # instead would incorrectly pack multiple groups together.
        batch_options=BatchOptions(max_bytes=2 * 16 * target_dtype.itemsize),
    )
    if direct:
        batch = MergeBatch(
            tuple(
                MergeBatch.from_tensors([model[name] for model in models]).groups[0]
                for name in models[0]
            )
        )
        tensors = method(batch, **options)
    else:
        tensors = merge_state_dicts(models, method, **options).values()
    assert len(calls) == 5
    assert all(ref() is None for ref in previous)
    for i, tensor in enumerate(tensors):
        torch.testing.assert_close(
            tensor, torch.full((16,), 2 * i + 4, dtype=torch.float16, device=device)
        )


@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_incremental_dtype_conversion_preserves_autograd(dtype, device):
    a = torch.ones(3, dtype=torch.float16, device=device, requires_grad=True)
    b = torch.ones(3, dtype=torch.bfloat16, device=device, requires_grad=True)
    result = merge_state_dicts(
        [{"w": a}, {"w": b}],
        "linear",
        parameters={"weight": [0.25, 0.75]},
        dtype=dtype,
        out_dtype=torch.float32,
    )["w"]
    result.sum().backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, 0.25))
    torch.testing.assert_close(b.grad, torch.full_like(b, 0.75))


@pytest.mark.parametrize("method_name", ["linear", "task_arithmetic"])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [None, torch.float64])
def test_direct_and_state_dict_calls_share_dtype_policy(method_name, dtype, out_dtype):
    models = mixed_models()
    parameters = {"weight": 0.5}
    expected = merge_state_dicts(
        models,
        method_name,
        base=0,
        parameters=parameters,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    names = [name for name in models[0] if name != "counter"]
    batch = MergeBatch(
        tuple(
            MergeBatch.from_tensors(
                [model[name] for model in models], base_index=0
            ).groups[0]
            for name in names
        )
    )
    actual = merge_methods.get(method_name)(
        batch,
        parameters=parameters,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    for name, tensor in zip(names, actual):
        torch.testing.assert_close(tensor, expected[name])


@pytest.mark.parametrize("option", ["dtype", "out_dtype"])
def test_direct_call_validates_dtype_before_execution(option):
    from mergekit.merge_methods import TensorGroup, merge_method

    @merge_method(name="never_execute")
    def kernel(group: TensorGroup) -> torch.Tensor:
        pytest.fail("Invalid execution settings reached the kernel")

    with pytest.raises(ValueError, match="floating-point torch.dtype"):
        kernel(MergeBatch.from_tensors([torch.ones(1)]), **{option: torch.int64})


def test_raw_graph_linear_cancellation_stays_finite(tmp_path, device):
    source = torch.tensor([-16.0, -1.0, 0.0, 0.5, 1.0, 16.0], dtype=torch.float16)
    models = []
    for index, weight in enumerate([1.0, -1.0, 1e-5]):
        path = tmp_path / f"model{index}.safetensors"
        save_file({"w": source}, path)
        models.append({"model": str(path), "parameters": {"weight": weight}})
    config = RawPyTorchMergeConfig(merge_method="linear", models=models)
    output = tmp_path / "output"
    tasks = plan_flat_merge(config, str(output), False, False, MergeOptions())
    Executor(tasks, math_device=device).execute()
    torch.testing.assert_close(load_file(output / "model.safetensors")["w"], source)
