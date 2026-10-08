> Historical block-cache stage. Superseded by the [current lifecycle findings](GPU_LIFECYCLE.md) and [frozen repeatability report](GPU_LIFECYCLE_REPEATABILITY.md). This report preserves the earlier negative results and baseline correction; its test count, commands and conclusions describe that stage. Use the [current GPU guide](GPU_COMPUTE.md) for the integrated implementation and reproduction instructions.

# Local elapsed-time optimization

Measured on RTX 5070 Ti with Torch 2.11.0+cu128, CUDA 12.8, width 2048,
batch cap 1024, 25 full Canterbury/Calgary files. See
`results/lossless/gpu_optimized.json` for all 21 alternating runs.

Changes: dense plans skip unused block hashes; blocks retain input buffer views;
packing uses a reusable contiguous NumPy buffer; identity references bypass GPU
gathering. A GPUCache retains distinct outputs across versions and invalidates
the index and outputs when kernel identity or width changes. Callers must supply
a new identity for every operation/configuration change. Cached input buffers
must remain immutable. The cache is currently unbounded and retains input views.

Latest cold medians in milliseconds:

| Workload | Dense | Reuse | Median paired dense minus reuse |
| --- | ---: | ---: | ---: |
| Single version | 15.8565 | 18.0430 | -3.0961 |
| Three versions | 49.2278 | 54.1520 | -5.2957 |

The original baseline single-version median was 20.3957 ms. These historical
measurements were separate runs: their difference is useful directional evidence,
not an alternating before/after experiment. The corrected dense baseline is
faster, so the earlier 17.76% reuse advantage does not establish an advantage
over this baseline.

Persistent-cache medians (initial construction kept separate):

| Version | Fresh dense | Cache | Median paired dense minus cache |
| --- | ---: | ---: | ---: |
| Initial original | 16.5115 | 18.9055 | -2.5880 |
| Insertion update | 16.4378 | 16.7847 | -0.5283 |
| Replacement update | 16.3237 | 16.4096 | 0.0347 |

An earlier pass showed update improvements of 6–9%, but these did not hold in
the latest pass. A reliable update advantage is not established.

Pinned-buffer/asynchronous-upload experiment: median paired savings were
0.8143 ms for single-version reuse and 0.4353 ms for three-version reuse.
Pinning and required staging-buffer reuse synchronization are included. This
remains opt-in; these small differences need repeat confirmation before changing
the default. All full outputs matched bitwise and all bytes reconstructed exactly
on every run. All 46 unit tests passed, including cache configuration invalidation.

Reproduce:

```powershell
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/benchmark_gpu.py data/corpora --repeats 21 --output results/lossless/gpu_optimized.json
```

Complete elapsed times include planning, packing, transfers, kernel execution,
gathering, full host outputs and byte reconstruction with integrity checks.
The FFT is cheap enough that CPU lookup and output materialization dominate.
Publication readiness remains on hold under the existing gate.
