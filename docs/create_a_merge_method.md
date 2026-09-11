# Defining Merge Methods

Merge methods are synchronous tensor operations callable from Python or through
MergeKit's YAML and raw-PyTorch merge commands.

## Group kernels

Group kernels operate directly on one logical output at a time, without allocating
packing buffers. Use them for simple methods, algorithms that need tensor metadata,
or work that does not benefit from vectorization:

```python
import torch

from mergekit.merge_methods import TensorGroup, merge_method, merge_tensors


@merge_method(name="scaled_copy")
def scaled_copy(group: TensorGroup, scale: float = 1.0) -> torch.Tensor:
    return group.tensors[0] * scale


result = merge_tensors([torch.ones(3)], scaled_copy, parameters={"scale": 2.0})
# tensor([2., 2., 2.])
```

Group kernels use ordinary annotations for shared Python values and `PerInput[T]`
or `PerNonBase[T]` for ordered mappings keyed by `TensorEntry.id`. Shared values
may include lists and other unhashable types; only batch kernels require hashable
Python options. Parameters are validated against their annotations, including
ordinary `Annotated` constraints such as `Annotated[float, Field(gt=0)]` and
constraints inside per-input `T`.

Group kernels receive mappings for per-input values; batch kernels receive tensors
annotated with `BatchParameter`. These annotations describe different runtime
types and cannot be interchanged. Unannotated parameters and tensor coefficients
without `BatchParameter` are rejected when defining the method.

`merge_method` selects execution from the first argument annotation: `TensorBatch`
for a numerical batch kernel, `TensorGroup` for a sequential group kernel. It can
also be called directly: `method = merge_method(kernel, name="my_method")`.
`method.supports_batching` reports whether the kernel is batched; accepting a logical
batch alone does not imply vectorization.

## A native batch kernel

```python
import torch
from typing import Annotated

from mergekit.merge_methods import (
    BatchParameter, ParameterScope, TensorBatch, merge_method,
)


@merge_method(name="weighted_average", pretty_name="Weighted Average")
def weighted_average(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: bool = True,
) -> torch.Tensor:
    # N borrowed tensors shaped [B, *weight_shape]; weight: [B, N], float32 or float64.
    first = batch.tensors[0]
    coefficient_shape = (first.shape[0],) + (1,) * (first.ndim - 1)
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    weight = weight.to(dtype)
    result = torch.zeros_like(first, dtype=dtype)
    for tensor, coefficient in zip(batch.tensors, weight.unbind(1)):
        result.addcmul_(tensor, coefficient.reshape(coefficient_shape))
    if normalize:
        denominator = weight.sum(1).reshape(coefficient_shape)
        if (denominator == 0).any():
            raise ValueError("Cannot normalize weights that sum to zero")
        result.div_(denominator)
    return result.to(first.dtype)  # [B, *weight_shape]
```

The function signature defines parameter names, types, scopes, and defaults:

- `Annotated[Tensor, BatchParameter(float)]` receives coefficients shaped `[B]`.
- `BatchParameter(float, ParameterScope.INPUT)` receives `[B, N]`.
- Adding `target=InputParameterTarget.NON_BASE` excludes the base input from the
  coefficient axis. Base-aware batches use a canonical base-first input layout.
- Ordinary annotations, such as `normalize: bool = True`, receive shared Python
  values. In batch kernels, each value is constant within a packed batch.
  Different option values partition work into separate batches. Batch options must
  be hashable (for example booleans, strings, enums, or tuples).
- A parameter without a default is required.

`BatchParameter` supports `float`, `int`, and `bool`, including constrained types
such as `PositiveFloat` or `Annotated[float, Field(ge=0, le=1)]`. Constraints belong
inside `BatchParameter(...)`; its enclosing annotation describes the kernel's
**Tensor** argument, not an individual coefficient. Float coefficients use
float32 unless the aligned inputs are float64, in which case they use float64.
Integer and boolean coefficients use int64 and bool. Kernels may explicitly cast
coefficients when choosing their intermediate precision.

Scopes do not encode the YAML hierarchy. For example, a shared parameter may still be
overridden per module or slice and may vary between output tensors through filters or
gradients. Scope describes the input axis along which the resolved value is bound.

Unsupported or ambiguous annotations are rejected when the method is defined. Resolved
values are validated against their annotations before any tensor math runs.

Configuration gradient endpoints are validated using the parameter's declared type
before interpolation. For example, `[1e-5, 1e-3]` interpolates for a float parameter
even when the YAML parser reads the endpoints as strings. String and boolean
parameters use discrete steps instead. Integer gradients must resolve to an integer;
fractional results and endpoints outside declared constraints are rejected.

## Input contracts

Structural requirements belong to the method definition:

```python
from mergekit.merge_methods import BasePolicy, InputContract


@merge_method(
    name="base_interpolation",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
def base_interpolation(
    batch: TensorBatch,
    t: Annotated[torch.Tensor, BatchParameter(float)],
) -> torch.Tensor:
    a, b = batch.tensors
    t = t.reshape(t.shape[0], *((1,) * (a.ndim - 1)))
    return (1 - t) * a + t * b
```

The available base policies are `IGNORED`, `OPTIONAL`, `REQUIRED`, and `FORBIDDEN`.
Input and non-base arities can be bounded independently. MergeKit validates contracts
while planning a configured merge and validates every group in a direct batch before
executing any group.

## Calling a method directly

Use `merge_tensors` for inputs contributing to one output tensor:

```python
from mergekit.merge_methods import merge_tensors

merged_tensor = merge_tensors(
    [tensor_a, tensor_b],
    "linear",
    parameters={"weight": [0.25, 0.75]},
)

interpolated = merge_tensors(
    [base_tensor, other_tensor],
    "slerp",
    base_index=0,
    parameters={"t": 0.3},
)
```

This always returns a tensor. It accepts a method name or a method object, optional
`ids` for mapping-valued per-input parameters, a `name` for tensor metadata and error
messages, and `dtype`/`out_dtype` controls. `base_index` refers to the input sequence,
even when custom IDs are supplied. It uses the same validation and execution as the
logical-batch interface below; checkpoint buffer handling belongs to `merge_state_dicts`.

The decorator constructs a callable method. Construction does not register it;
registration is only needed for lookup by name. Built-ins are already registered.
For multiple outputs or explicit tensor metadata, pass a sequence of groups:

```python
from mergekit import merge_methods
from mergekit.merge_methods import BatchOptions, PerGroupValues, TensorGroup

groups = [
    TensorGroup.from_tensors(
        [a_attention, b_attention],
        ids=["a", "b"],
        name="attention.weight",
    ),
    TensorGroup.from_tensors(
        [a_embedding, c_embedding],
        ids=["a", "c"],
        name="embedding.weight",
        is_embed=True,
    ),
]

attention, embedding = merge_methods.get("linear")(
    groups,
    parameters={
        "weight": PerGroupValues([
            {"a": 0.25, "b": 0.75},
            {"a": 0.50, "c": 0.50},
        ]),
    },
    batch_options=BatchOptions(max_bytes=64 * 1024**2),
)
```

Sequences may be used for per-input values when their order matches the tensor entries.
Mappings are preferable whenever the inputs already have stable IDs. Scalar per-input
defaults and scalar arguments are broadcast.

Modules and state dictionaries also have a convenience adapter:

```python
from mergekit.merge_methods import merge_state_dicts

merged = merge_state_dicts(
    {"a": model_a, "b": model_b},
    "linear",
    parameters={"weight": {"a": 0.25, "b": 0.75}},
)
```

By default this validates that all state dictionaries have the same keys before
executing the batch. Pass `strict=False` to merge their intersection.
Non-floating buffers, such as BatchNorm counters, must agree exactly across inputs;
they are copied without numerical merging. Differing buffers raise an error.

Module inputs retain their parameter autograd connections, just like dictionaries
of live tensors. Use `torch.no_grad()` when you do not need to record merge operations.

`merge_state_dicts` promotes floating inputs independently for each weight. Matching
bfloat16 inputs remain bfloat16; float16 with bfloat16 promotes to float32, as does
bfloat16 with float32. Float64 inputs promote the group to float64. Pass
`dtype=torch.bfloat16` to explicitly cast inputs, or `out_dtype=torch.bfloat16` to
cast only the merged outputs. Both options leave non-floating buffers unchanged.
The YAML and raw-PyTorch adapters use the corresponding string-valued `dtype` and
`out_dtype` settings. `dtype` selects the input representation used for merging;
adapters may apply this explicit cast during loading, before device transfer, to
reduce memory and transfer costs.
Both `merge_tensors` and direct method calls accept the same dtype options and
use the same promotion policy. Algorithm parameters go in the `parameters` mapping,
separately from dtype and packing controls; no algorithm parameter names are reserved.

## Batches

A `TensorGroup` contains the inputs for one logical output tensor. Pass a list or
tuple of groups to a method; it returns a tuple of tensors with one output per
group, in input order. Unpack a singleton result with
`(merged_tensor,) = method([group], ...)`. An empty sequence returns an empty tuple.

`TensorGroup.from_tensors(tensors, ids=..., base_index=..., name=..., is_embed=...)`
borrows the supplied tensors. IDs default to integer positions; `base_index` always
refers to the supplied tensor order. Groups may use different IDs, input counts,
shapes, and metadata. The `TensorGroup(entries=..., metadata=...)` constructor
accepts existing `TensorEntry` and `TensorMetadata` objects.

Logical batches may be heterogeneous. Preparation validates every group, buckets
compatible work by shape, dtype, device, input count/layout, and execution options,
then prepares numerical `TensorBatch` inputs one chunk at a time. Singleton chunks
borrow an `unsqueeze(0)` view of each input, copying only for dtype conversion.
Larger chunks stack outputs separately for each input. All inputs within a group
must have matching shapes and devices. Dtypes are aligned during execution,
including for sequential group kernels. Every group is checked before any kernel executes, and kernels must preserve
the weight shape. Output order always matches logical group order, not bucket order.
Dense strided inputs need not be contiguous; kernels reshape locally where needed.

Merge methods do not truncate embeddings or repair incompatible tensors. Configure
`tokenizer: {source: base}` to align inputs to the base vocabulary, or select `union`
or a specific model's tokenizer. Vocabulary alignment, missing-token initialization,
and padding happen before merging; hidden dimensions must already match. Direct
tensor and state-dict callers must perform any alignment themselves.

Kernel execution disables ambient autocast so input dtype and method-specific
precision choices determine the arithmetic. Linear and SLERP use float32 for
float16, bfloat16, and float32 inputs, and float64 for float64 inputs. Linear
accumulates into one buffer and normalizes before casting back to the input dtype.
A zero coefficient sum in the working precision is rejected when normalization is
enabled; nearly cancelling weights can lose accuracy. CPU execution may additionally
cast the current input internally; no full-precision copy of every input is retained.

The kernel receives a tuple of N tensors shaped `[B, *weight_shape]` and returns
one tensor shaped `[B, *weight_shape]`. It must
preserve the output axis: reductions for norms, means, and dot products must not
accidentally combine different outputs. Linear and SLERP have native batched
implementations.
SLERP processes weights in chunks to bound full-precision inference scratch.
Source tensors, packed inputs, outputs, and retained autograd graphs require
additional storage.

Ordinary parameters broadcast across groups. Use `PerGroupValues` to explicitly vary
a parameter along the outer batch axis; the wrapper avoids ambiguity with shared
list-valued parameters.

```python
from mergekit.merge_methods import BatchOptions, PerGroupValues

result = merge_methods.get("linear")(
    groups,
    parameters={"weight": PerGroupValues([weights_for_group_0, weights_for_group_1])},
    batch_options=BatchOptions(max_bytes=64 * 1024 * 1024, max_groups=128),
)
```

The same `batch_options` argument is accepted by `merge_state_dicts`. Limits apply
to packed input and coefficient buffers, **not** source tensors, retained outputs,
autograd graphs, or kernel scratch space. A single oversized group executes alone;
these are packing limits, not a guarantee of total GPU memory usage.
The common method call performs any remaining input conversion only for the current
chunk (or current group for sequential methods), and casts outputs to `out_dtype`
before retaining them. An input already cast by its loader needs no further conversion.
Packing budgets use the target input dtype, including when inputs are promoted.
Autograd graphs and outputs that alias converted inputs can retain that storage.

## Ownership and low-level execution

All inputs are borrowed and must not be modified, including inputs assembled by
the batch preparer. They may have arbitrary strides. Use `tensor.clone()` for
writable storage, `tensor.contiguous()` when an individual input must be contiguous,
or `torch.stack(batch.tensors, dim=1)` when a kernel needs a single allocation
shaped `[B, N, *weight_shape]`. Contiguity does not grant permission to modify an
input: `contiguous()` may return the original tensor. These allocations belong to
the kernel and count as scratch, outside packing limits. Outputs may alias inputs
or working storage, extending their lifetime.

A loader or executor that already has aligned buffers can call
`method.merge_batch(TensorBatch(...), **aligned_parameters)` directly. This is the
numerical boundary: callers supply correctly shaped coefficient tensors on the
input device and resolved Python options, and are responsible for satisfying the
method's input contract. It does not repeat logical parameter binding. For example,
`TensorBatch((a.unsqueeze(0), b.unsqueeze(0)), base_index=0)` represents one output
with two borrowed inputs.

## Metadata and base inputs

`TensorEntry.id` is an opaque, non-`None` hashable identifier (`None` is reserved for
the absence of a base). The core method API does not require
`ModelReference`; direct callers can use strings or integer positions.

`TensorGroup.metadata` carries lightweight output information such as its name and
whether it is an embedding tensor. A base is represented by an entry with
`is_base=True`, accessible through `group.base` and excluded from `group.non_base`.

## Registration

Method construction has no registration side effects. To make a method available
by name, register the constructed object explicitly:

```python
from mergekit.merge_methods import get, register, registered_methods

register(weighted_average)
assert get("weighted_average") is weighted_average
```

`register()` rejects duplicate names. `registered_methods()` returns a tuple of
currently registered methods. Passing a method object to `merge_state_dicts` or
calling it directly never requires registration.

All built-ins are assembled in `mergekit/merge_methods/registry.py`. Add the method
import and its object to that module's registration sequence. Class-based families
such as TIES/DARE use the same `register()` function as function-based methods.

Read method metadata directly from `method.spec.name`, `method.spec.pretty_name`,
and `method.spec.reference_url`.

## Explicit specifications and class-based methods

The decorator is the usual authoring path, but signature inference is optional.
`GroupMergeMethod` subclasses can provide a `MergeMethodSpec` and implement
`merge_group(group, **parameters)`. A native numerical kernel can instead be
wrapped with `BatchedMergeMethod(spec, implementation)`. Both use the same group
sequence interface, validation, dtype handling, and registration.

Use explicit `ParameterSpec` objects from `mergekit.merge_methods.base` when a
method's parameters depend on its construction options or cannot be expressed in
a Python signature. The built-in
[`GeneralizedTaskArithmeticMerge`](../mergekit/merge_methods/generalized_task_arithmetic.py)
uses this path for its sparsification-dependent parameters and its `"lambda"`
parameter.

## Execution adapters

Direct callers use `MergeMethod.__call__` to validate and bind parameters before
execution. The computation graph uses a generic `ExecuteMergeMethodTask`, built
with `from_parameters()` from already-resolved planner settings. It binds per-input
values to stable integer positions during planning and uses the same internal
execution path without repeating scalar validation or parameter binding. Missing
optional inputs retain their original positions and coefficients; loaded tensor
shapes, devices, and input contracts are still checked at execution time. Singleton
batches borrow input views without compatibility bucketing or packing.

The YAML and raw-PyTorch planners use the same parameter resolver. Each adapter
supplies settings in descending precedence; the resolver applies filters, gradients,
validated defaults, requiredness, and input/base targeting from the method spec.
Shared parameters match the output tensor name; per-input parameters match each
source tensor's name. An implicitly added base receives all-input parameters from
the same global settings/defaults as an explicit input.

Methods may set `uses_accelerator=False` when constructing or registering their
spec to execute on the storage device instead of requesting a transfer to the math
device. Passthrough uses this for both plain copies and scaled copies. This is a
graph scheduling hint; direct calls always operate on the supplied devices.

The YAML graph adapter submits one logical output at a time. State-dict callers
batch compatible outputs. The graph adapter handles missing optional weights
according to the method spec's optional-tensor policy. Outside those fallbacks,
base-aware methods require a configured base to be present: a missing base weight
must not silently select a baseless variant of the algorithm.
