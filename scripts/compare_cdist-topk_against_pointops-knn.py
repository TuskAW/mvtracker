import time

import torch
from pointops import knn_query

B, N, M, D, K = 12, 49152, 928, 3, 16


def knn_torch(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    dists = torch.cdist(xyz_query, xyz_ref, p=2)  # shape: (B, M, N)
    sorted_dists, indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    return sorted_dists, indices


def knn_pointops(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    B, N, _ = xyz_ref.shape
    _, M, _ = xyz_query.shape
    orig_dtype = xyz_ref.dtype

    xyz_ref_flat = xyz_ref.contiguous().view(B * N, 3).to(torch.float32)
    xyz_query_flat = xyz_query.contiguous().view(B * M, 3).to(torch.float32)

    offset = torch.arange(1, B + 1, device=xyz_ref.device) * N
    new_offset = torch.arange(1, B + 1, device=xyz_query.device) * M

    idx, dists = knn_query(k, xyz_ref_flat, offset, xyz_query_flat, new_offset)

    # Remap global indices to local per-batch
    idx = idx.view(B, M, k)
    idx = idx - (torch.arange(B, device=idx.device).view(B, 1, 1) * N)
    dists = dists.view(B, M, k).to(orig_dtype)

    return dists, idx


def benchmark(fn, name, HALF_PRECISION=False, iters=100):
    total_time = 0.0
    peak_memories = []
    for _ in range(iters):
        xyz_ref = torch.randn(B, N, D, device="cuda")
        xyz_query = torch.randn(B, M, D, device="cuda")
        if HALF_PRECISION:
            xyz_ref = xyz_ref.half()
            xyz_query = xyz_query.half()
        fn(K, xyz_ref, xyz_query)  # warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        fn(K, xyz_ref, xyz_query)
        torch.cuda.synchronize()
        total_time += time.time() - start
        peak_memories.append(torch.cuda.max_memory_allocated() / 1e6)  # MB

    avg_time = total_time / iters
    peak_memory_min = min(peak_memories)
    peak_memory_avg = sum(peak_memories) / len(peak_memories)
    peak_memory_max = max(peak_memories)
    print(f"{name:<24} | "
          f"Avg Time: {avg_time:.6f} s | "
          f"Peak Memory: {peak_memory_avg:>6.2f} MB (min: {peak_memory_min:>6.2f}, max: {peak_memory_max:>6.2f})")


print("Benchmarking KNN with different methods (HALF_PRECISION=True):")
benchmark(knn_torch, "torch.cdist+torch.topk", True)
benchmark(knn_pointops, "pointops.knn_query", True)

print("\nBenchmarking KNN with different methods (HALF_PRECISION=False):")
benchmark(knn_torch, "torch.cdist+torch.topk", False)
benchmark(knn_pointops, "pointops.knn_query", False)
