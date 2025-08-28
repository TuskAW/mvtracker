# Profiling Notes

This document summarizes how to run performance profiling using PyTorch’s built-in tools, and how to interpret the results.

To profile one training iteration (forward + backward + optimizer step), the following snippet can be used:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    with_flops=True,
    profile_memory=True,
    record_shapes=True,
) as prof:
    # one iteration of fwd + bwd + optimize
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=36))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=36))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=36))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=36))

prof.export_chrome_trace("trace.json")
breakpoint()
```

The printed summary tables produced by `.key_averages()` are already extremely informative. Sorting by `cuda_time_total` highlights which operations dominate the runtime on GPU. The `self_cuda_memory_usage` sort can reveal the main contributors to memory consumption. Enabling `record_shapes=True` further helps diagnose operations that unexpectedly receive large tensors.

For more detailed inspection, the exported trace file `trace.json` can be opened in Chrome at `chrome://tracing`. This presents a flamegraph-style timeline of kernel launches, memory activity, and execution order, which can be especially helpful for understanding the global structure and scheduling behavior of the code. For a brief tutorial on how to navigate this view, see [here](https://www.youtube.com/watch?v=AhIOohJYSrw).

Note that the trace file can become very large (e.g., over 1 GB) depending on how much code is profiled. This may slow down both trace generation and visualization. While `.key_averages()` provides a fast, summary-level view that is often sufficient for identifying key bottlenecks, the flamegraph timeline can be equally valuable for temporal insights.