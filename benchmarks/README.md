# Merge compute benchmarks

Run from the repository root in the project environment:

```sh
python benchmarks/merge_methods.py --device cpu --threads 1
python benchmarks/merge_methods.py --device cuda
python benchmarks/merge_batching.py --device cuda --shape 32 --groups 1 256
python benchmarks/merge_batching.py --device cuda --shape 2048 2048 --groups 1 4
```

`merge_methods.py` measures `ExecuteMergeMethodTask.execute()` with already-loaded
inputs. Planning (including parameter resolution and binding), graph scheduling,
loading, tokenizer alignment, and writing are excluded. The direct-kernel comparison
uses borrowed singleton views and excludes adapter overhead.

`merge_batching.py` compares repeated singleton graph calls with one
`merge_state_dicts` call. Both paths retain all outputs and include tensor validation
and any packing. Singleton inputs are borrowed; compatible outputs in larger
batches are stacked separately for each input. Graph parameters are bound before
timing; the state-dict API binds parameters on each invocation. The script checks
that their outputs agree before measuring. Use `--max-packed-mib` to vary the
state-dict packing budget (64 MiB by default).

CUDA peak allocation includes returned outputs and scratch, but excludes source
tensors. It does not measure allocator-reserved memory or process-wide GPU memory.
Run benchmarks without concurrent tests or GPU workloads.

Packing limits exclude kernel scratch and returned outputs, so batching multiple
outputs can increase peak memory. For dtype and storage requirements, see
[Defining Merge Methods](../docs/create_a_merge_method.md#batches).
