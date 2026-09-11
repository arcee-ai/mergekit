# Defining Merge Methods

Merge methods are ordinary, synchronous tensor operations. They do not depend on
MergeKit's configuration format, model loaders, or computation graph. Those systems
adapt their inputs to the same callable method interface used by in-memory callers.

## A native batch kernel

```python
import torch
from typing import Annotated

from mergekit.merge_methods import (
    BatchParameter, Option, ParameterScope, TensorBatch, merge_method,
)


@merge_method(name="weighted_average", pretty_name="Weighted Average")
def weighted_average(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: Option[bool] = True,
) -> torch.Tensor:
    # tensors: [B, N, *weight_shape], weight: [B, N]
    tensors = batch.tensors
    weights = weight.reshape(*weight.shape, *((1,) * (tensors.ndim - 2)))
    result = (tensors * weights).sum(1)
    if normalize:
        result = result / weights.sum(1)
    return result.to(tensors.dtype)  # [B, *weight_shape]
```

The function signature is the single source of truth for parameter names, types,
scopes, defaults, and requiredness:

- `Annotated[Tensor, BatchParameter(float)]` receives coefficients shaped `[B]`.
- `BatchParameter(float, ParameterScope.INPUT)` receives `[B, N]`.
- Adding `target=InputParameterTarget.NON_BASE` excludes the base input from the
  coefficient axis. Base-aware batches use a canonical base-first input layout.
- `Option[T]` receives a Python value that is constant within a packed batch.
  Different option values partition work into separate batches. Options must be
  hashable (for example booleans, strings, enums, or tuples).
- A parameter without a default is required.

`BatchParameter` supports `float`, `int`, and `bool`, including constrained types
such as `PositiveFloat` or `Annotated[float, Field(ge=0, le=1)]`. Constraints belong
inside `BatchParameter(...)`; its enclosing annotation describes the kernel's
**Tensor** argument, not an individual coefficient. Float coefficients use float32
(float64 for float64 inputs); integer and boolean coefficients use int64 and bool.
Annotations therefore describe the actual runtime kernel types without pretending
that a batched coefficient is still a Python float.

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
    a, b = batch.tensors[:, 0], batch.tensors[:, 1]
    t = t.reshape(t.shape[0], *((1,) * (a.ndim - 1)))
    return (1 - t) * a + t * b
```

The available base policies are `IGNORED`, `OPTIONAL`, `REQUIRED`, and `FORBIDDEN`.
Input and non-base arities can be bounded independently. MergeKit validates contracts
while planning a configured merge and validates every group in a direct batch before
executing any group.

## Calling a method directly

The decorator constructs a callable method. Construction does not register it;
registration is only needed for lookup by name. Built-ins are already registered:

```python
from mergekit import merge_methods
from mergekit.merge_methods import MergeBatch

batch = MergeBatch.from_tensors(
    [tensor_a, tensor_b],
    ids=["model_a", "model_b"],
)

result = merge_methods.get("linear")(
    batch,
    parameters={"weight": {"model_a": 0.25, "model_b": 0.75}},
)
merged_tensor = result.one()
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

`merge_state_dicts` promotes floating inputs independently for each weight. Matching
bfloat16 inputs remain bfloat16; float16 with bfloat16 promotes to float32, as does
bfloat16 with float32. Float64 inputs promote the group to float64. Pass
`dtype=torch.bfloat16` to explicitly cast inputs, or `out_dtype=torch.bfloat16` to
cast only the merged outputs. Both options leave non-floating buffers unchanged.
The YAML and raw-PyTorch adapters use the corresponding string-valued `dtype` and
`out_dtype` settings. Raw-PyTorch merges also copy equal non-floating buffers
without casting them, and reject differing buffers before dtype conversion.
Direct `MergeBatch` calls accept the same dtype options and
use the same promotion policy. Algorithm parameters go in the `parameters` mapping,
separately from dtype and packing controls; no algorithm parameter names are reserved.

## Batches

A `TensorGroup` contains the inputs for one logical output tensor. A `MergeBatch`
contains one or more groups, and invoking a method returns a `MergedBatch` with one
output per group.

Logical batches may be heterogeneous. Preparation validates every group, buckets
compatible work by shape, dtype, device, input count/layout, and execution options,
then incrementally packs numerical `TensorBatch` buffers. All inputs within a group
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
precision choices determine the arithmetic. Linear normalizes its small coefficient
array in float32 (float64 for float64 inputs), then casts coefficients to the input
dtype for its matrix product. It does not allocate a full float32 input copy.
Coefficient rounding and backend accumulation affect low-precision results; use
float32 inputs when that additional precision is needed.

The kernel receives `[B, N, *weight_shape]` and returns `[B, *weight_shape]`. It must
preserve the output axis: reductions for norms, means, and dot products must not
accidentally combine different outputs. Linear and SLERP have native batched
implementations; SLERP stays entirely in Torch on the input device.
SLERP reduces norms and dot products in chunks and writes its output in chunks,
bounding full-precision inference scratch even for oversized individual weights.
It selects interpolation coefficients per output before applying them to weights,
without materializing both spherical and linear results. Source tensors, packed
inputs, outputs, and retained autograd graphs still require their own storage.

Ordinary parameters broadcast across groups. Use `PerGroupValues` to explicitly vary
a parameter along the outer batch axis; the wrapper avoids ambiguity with shared
list-valued parameters.

```python
from mergekit.merge_methods import BatchOptions, PerGroupValues

result = merge_methods.get("linear")(
    logical_batch,
    parameters={"weight": PerGroupValues([weights_for_group_0, weights_for_group_1])},
    batch_options=BatchOptions(max_bytes=64 * 1024 * 1024, max_groups=128),
)
```

The same `batch_options` argument is accepted by `merge_state_dicts`. Limits apply
to packed input and coefficient buffers, **not** source tensors, retained outputs,
autograd graphs, or kernel scratch space. A single oversized group executes alone;
these are packing limits, not a guarantee of total GPU memory usage.
The common method call converts inputs only for the current chunk (or current group
for sequential methods), and casts outputs to `out_dtype` before retaining them.
Packing budgets use the target input dtype, including when inputs are promoted.
Autograd graphs and outputs that alias converted inputs can retain that storage.

## Ownership and low-level execution

Logical inputs are borrowed and must not be modified. Packing creates owned working
storage. `batch.workspace()` returns that storage for an owned batch, or clones it
for a borrowed batch. Kernels may use it for in-place work when compatible with
their autograd requirements. Outputs may alias working storage, extending its
lifetime; a group method may also return a borrowed input unchanged.

A loader or executor that already has aligned buffers can call
`method.merge_batch(TensorBatch(...), **aligned_parameters)` directly. This is the
numerical boundary: callers supply correctly shaped coefficient tensors on the
input device and resolved Python options, and are responsible for satisfying the
method's input contract. It does not repeat logical parameter binding. Set
`owned=True` only when transferring permission to overwrite the buffer.

## Group kernels

Group kernels operate directly on one logical output at a time, without allocating
packing buffers. They are an alternative to native batch kernels, particularly for
algorithms that need tensor metadata or do not benefit from vectorization:

```python
from mergekit.merge_methods import Shared, TensorGroup, merge_method

@merge_method(name="scaled_copy")
def scaled_copy(group: TensorGroup, scale: Shared[float] = 1.0) -> torch.Tensor:
    return group.entries[0].tensor * scale
```

Group kernels use `Shared[T]`, `PerInput[T]`, and `PerNonBase[T]`. Per-input values
are ordered mappings keyed by `TensorEntry.id`. The signature remains the source
of parameter validation, including constraints inside `T`.

`merge_method` selects execution from the first argument annotation: `TensorBatch`
for a numerical batch kernel, `TensorGroup` for a sequential group kernel. It can
also be called directly: `method = merge_method(kernel, name="my_method")`.
`method.supports_batching` reports whether the kernel is batched; accepting a logical
batch alone does not imply vectorization.

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
and `method.spec.reference_url`; there are no separate metadata accessor methods.

## Execution adapters

Every adapter executes through `MergeMethod.__call__`, which validates inputs and
parameters, selects dtypes, and dispatches to the group or batch strategy. The
computation graph uses a generic `ExecuteMergeMethodTask`; individual methods do
not define tasks. The YAML planner resolves configured values and builds that adapter.
Other consumers can construct `MergeBatch` directly without importing graph or config
types.

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

The current YAML graph adapter still submits one logical output at a time; it does
not yet coalesce graph tasks. State-dict callers already batch compatible outputs.
The numerical boundary allows a future scheduler to load directly into packed
buffers without changing kernels. Missing-optional-weight fallback is handled by
the graph adapter through the method spec's optional-tensor policy, not by weakening
the numerical kernel's arity contract. Outside those explicit fallbacks, base-aware
methods require a configured base to be present: a missing base weight must not
silently select a baseless variant of the algorithm.
