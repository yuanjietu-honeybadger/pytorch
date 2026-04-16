# Inductor Triton Kernel Launch Overhead: Analysis & Optimization

## Overview

This document describes the overhead sources in PyTorch Inductor's Triton kernel
launch path and how D100887242 + D101027739 eliminate them, reducing per-kernel
launch latency from ~9µs to ~3µs (within 1µs of bare `cuLaunchKernel`).

## Architecture: How Inductor Launches a Triton Kernel

When Inductor compiles a model, each Triton kernel call goes through this stack:

```
Python generated code (inductor output)
  └─ CachingAutotuner.run()              ← Layer 3: autotuner dispatch
       └─ launcher(*args, stream=...)     ← generated function (per kernel)
            └─ runner(grid, ..., stream, *args)
                 └─ _StaticCudaLauncher._launch_kernel(...)   ← Layer 2: C extension
                      └─ cuLaunchKernel(...)                  ← Layer 1: CUDA driver
```

## Overhead Breakdown (Before Optimization)

Measured on GB200, @mode/opt, nop kernel with 19 tensor args:

```
Total per-kernel overhead:        ~9.06µs
├── Layer 3: CachingAutotuner.run()  ~2.0µs
│   ├── get_active_debug_mode()         ~0.1µs  (always None)
│   ├── triton.set_allocator()          ~0.3µs  (idempotent, never changes)
│   ├── TritonBundler.put_winner()      ~0.1µs  (no-op in production)
│   ├── len(self.launchers) != 1        ~0.1µs  (always False after 1st call)
│   ├── combo_tuning / coordesc checks  ~0.2µs  (always False)
│   ├── tuple(args) copy                ~0.5µs  (only needed for profiler)
│   └── dump_launch_params/tensors      ~0.2µs  (always False)
│
├── Layer 2: _StaticCudaLauncher._launch_kernel()  ~4.9µs
│   ├── PyArg_ParseTuple("KiiiiisOK")   ~0.3µs  (re-parses 9 fixed args)
│   ├── cuCtxGetCurrent validation       ~0.1µs  (redundant after 1st call)
│   ├── getPointer() per tensor arg:     ~0.3µs × N
│   │   ├── PyObject_GetAttrString("data_ptr")  → attr lookup
│   │   ├── PyObject_Call(data_ptr, ())          → Python call
│   │   └── cuPointerGetAttribute(DEVICE_PTR)    → driver validation
│   └── parseKernelArgs type-switch loop ~0.2µs
│
└── Layer 1: cuLaunchKernel              ~2.1µs  (irreducible driver cost)
```

### Why These Costs Exist

The existing `_StaticCudaLauncher._launch_kernel()` was designed as a
general-purpose entry point:

1. **`PyArg_ParseTuple("KiiiiisOK")`** — Parses ALL arguments from a Python
   tuple every call: CUfunction, gridX/Y/Z, numWarps, sharedMem, argTypes
   string, args tuple, stream. But kernel metadata (CUfunction, numWarps,
   sharedMem, argTypes) is *fixed* after compilation — only grid, stream,
   and kernel args change per call.

2. **`cuCtxGetCurrent` validation** — Checks that a CUDA context exists.
   Needed on the first call, but the context is guaranteed valid for all
   subsequent calls within the same process.

3. **`getPointer()` per tensor arg** — The slow path for extracting a device
   pointer from a Python tensor:
   ```cpp
   // Existing path: 3 operations per tensor
   auto ptr = PyObject_GetAttrString(obj, "data_ptr");  // Python attr lookup
   auto ret = PyObject_Call(ptr, empty_tuple, nullptr);  // Python method call
   cuPointerGetAttribute(&dev_ptr, DEVICE_PTR, data_ptr);// driver validation
   ```
   The `cuPointerGetAttribute` call asks the CUDA driver to validate that the
   pointer is a valid device pointer and return its device-mapped address.
   This is redundant for tensors allocated by PyTorch — they are *always*
   valid CUDA device pointers.

## Solution

### D100887242: Fast Path for CachingAutotuner.run() (Layer 3)

**Insight**: After the first successful kernel launch in steady state, all the
preamble checks in `CachingAutotuner.run()` are guaranteed to be no-ops.
We cache the launcher function and skip directly to it.

```python
# Hot path (added):
fast = self._fast_launcher
if (fast is not None
    and not benchmark_run and not kwargs
    and not autograd_profiler._is_profiler_enabled
    and not get_active_debug_mode()):
    return fast(*args, stream=stream)

# Cold path (existing): full preamble...
```

**Safety**: The fast path is only populated when ALL conditions hold (single
launcher, no debug, no profiler, no interpret, no dump). Profiler and debug
mode are re-checked every call to support runtime enable/disable.

**Savings**: ~2µs per kernel launch (eliminates entire Layer 3 preamble).

### D101027739: _FastCudaLauncher Vectorcall C Extension (Layer 2)

**Insight**: For the C extension layer, we can eliminate all redundant work
by pre-binding fixed kernel metadata at construction time and using the
fastest possible Python→C calling convention.

#### Pre-binding (Construction Time, Once Per Kernel)

```cpp
struct FastCudaLauncherObject {
    PyObject_HEAD
    vectorcallfunc vectorcall;     // PEP 590 vectorcall slot
    CUfunction func;               // pre-bound: never changes
    uint32_t numWarps;             // pre-bound
    uint32_t sharedMemBytes;       // pre-bound
    int numKernelArgs;             // pre-bound
    char argTypes[MAX_ARGS + 1];   // pre-bound: "OOOlOOO..."
    uint64_t argStorage[MAX_ARGS]; // pre-allocated scratch
    void* kernelArgs[MAX_ARGS];    // pre-computed: &argStorage[i]
};
```

Everything that was re-parsed by `PyArg_ParseTuple` on every call is now
stored once. The `kernelArgs[i] = &argStorage[i]` indirection pointers are
also pre-computed.

#### Vectorcall (PEP 590): Zero-Overhead Python→C Dispatch

Standard Python calling convention (`tp_call`) allocates a tuple for positional
args on every call. PEP 590 vectorcall passes args as a C array directly:

```
tp_call path:   Python → PyTuple_New → tp_call(self, args_tuple, kwargs)
                         ↑ heap allocation

vectorcall:     Python → vectorcall(self, args_array, nargs, kwnames)
                         ↑ stack pointer, no allocation
```

By setting `Py_TPFLAGS_HAVE_VECTORCALL` and the `vectorcall` offset,
CPython's call machinery skips tuple creation entirely.

#### THPVariable_Unpack: Zero-Cost Tensor Pointer Extraction

The biggest per-arg win. Compare:

```
BEFORE (getPointer):                    AFTER (getPointerFast):
├── PyObject_GetAttrString("data_ptr")  ├── THPVariable_Unpack(obj)
│   → hash table lookup, descriptor     │   → reinterpret_cast<THPVariable*>
│     protocol, bound method creation   │     → .cdata (C++ at::Tensor ref)
├── PyObject_Call(ptr, (), NULL)        │     Cost: 0 — just a pointer cast
│   → function call overhead            │
├── cuPointerGetAttribute(...)          ├── .data_ptr()
│   → CUDA driver round-trip            │   → inline: storage offset math
│                                       │     Cost: ~2ns
│   Total: ~300ns per tensor            │   Total: ~5ns per tensor
```

`THPVariable_Unpack` is a `reinterpret_cast` that accesses the `at::Tensor`
stored inside `THPVariable` — the same C++ object that backs every Python
`torch.Tensor`. Since we *know* the argument is a tensor (from `argTypes`),
we skip all type checking and pointer validation. The `data_ptr()` call is
just `storage_offset + storage.data()` — pure pointer arithmetic.

The `cuPointerGetAttribute` call is eliminated entirely. This is safe because:
- PyTorch tensors are always allocated via `cudaMalloc`/`cudaMallocAsync`
- The CUDA context is guaranteed valid (checked once at kernel load time)
- The pointer is guaranteed to be a device pointer (not host/managed)

#### No cuCtxGetCurrent Check

The existing path calls `cuCtxGetCurrent` on every launch to verify the CUDA
context exists. `_FastCudaLauncher` skips this because:
- The CUfunction was loaded via `cuModuleLoadData` which requires a valid context
- If the context were destroyed, the CUfunction would be invalid anyway
- No production code destroys CUDA contexts mid-run

## Results

### Microbenchmark (tritonbench launch_latency, GB200)

| Stage | Before | After D100887242 | After D101027739 |
|-------|--------|-------------------|-------------------|
| Per-kernel (0-arg) | 6.85µs | 5.72µs (-16%) | **3.12µs (-54%)** |
| Per-kernel (19-arg) | 9.06µs | 6.31µs (-30%) | **3.28µs (-64%)** |
| Raw cuLaunchKernel | 2.09µs | 2.09µs | 2.09µs |
| Gap to driver | 4.76µs | 3.63µs | **1.03µs** |

The 19-arg case improved the most because `getPointer()` × 19 was the
dominant cost. With `THPVariable_Unpack`, tensor count barely matters
(3.28µs vs 3.12µs = only 0.16µs for 19 tensors, ~8ns each).

### E2E Model Benchmark (torchbench, inductor, no cudagraphs, bfloat16, GB200)

| Model | Kernels | Baseline | Optimized | Δ |
|-------|---------|----------|-----------|---|
| hf_Bert | 285 | 2.513ms | 1.907ms | **-24.1%** |
| resnet18 | 69 | 0.894ms | 0.601ms | **-32.8%** |
| resnet50 | 175 | 2.242ms | 2.227ms | ~0% (compute-bound) |
| mobilenet_v2 | 153 | 2.537ms | 1.147ms | **-54.8%** |

Models with many small kernels see the largest improvements. Compute-bound
models (resnet50) are unaffected — the optimization has zero cost when kernel
execution time dominates launch overhead.

### With CUDAGraphs

| Model | CG Baseline | CG + Optimized | Δ |
|-------|-------------|----------------|---|
| hf_Bert | 1.777ms | 1.739ms | ~0% |
| resnet18 | 0.552ms | 0.545ms | ~0% |
| resnet50 | 2.193ms | 2.220ms | ~0% |
| mobilenet_v2 | 0.952ms | 0.954ms | ~0% |

CUDAGraphs replays a recorded command buffer, completely bypassing the Python
dispatch stack. The optimization correctly has no effect in this mode.

## Summary of Techniques

| Technique | What it eliminates | Savings |
|---|---|---|
| Fast-path cache in `CachingAutotuner.run()` | Python preamble (debug, allocator, bundler, combo checks) | ~2µs/kernel |
| Pre-bound kernel metadata | `PyArg_ParseTuple` re-parsing 9 fixed args per call | ~0.3µs/kernel |
| PEP 590 vectorcall | Tuple allocation for positional args | ~0.1µs/kernel |
| `THPVariable_Unpack` + `data_ptr()` | `PyObject_GetAttrString` + `PyObject_Call` + `cuPointerGetAttribute` per tensor | ~0.3µs × N tensors |
| Skip `cuCtxGetCurrent` | Redundant context validation | ~0.1µs/kernel |
