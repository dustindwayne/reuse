> Historical record. Superseded by the [frozen lifecycle findings](../GPU_LIFECYCLE.md). Older conclusions remain here to preserve the experiment trail.

# Local elapsed-time optimization: block-cache stage

Provenance: supplied local-run report, received 2026-10-08. This session has not inspected the raw timings or local implementation. The later [chunk-cache report](GPU_CHUNK_LOOKUP.md) supersedes the update conclusions below. These changes and measurements were not committed or pushed from the desktop when reported; this document records the findings, not an integrated implementation.

Hardware: RTX 5070 Ti, Torch 2.11.0+cu128, CUDA 12.8. Width 2048, batch cap 1024, all 25 full Canterbury/Calgary files. Local raw measurements: `results/lossless/gpu_optimized.json`, 21 alternating runs. That JSON is not included in this commit.

## Changes reported

Dense plans skip unused block hashes. Blocks retain input buffer views. Packing uses a reusable contiguous NumPy buffer. Identity references bypass GPU gathering. A GPUCache retains distinct outputs across versions and invalidates the index and outputs when kernel identity or width changes. Callers supply a new identity for every operation/configuration change. Cached input buffers remain immutable. Cache capacity is unbounded and input views are retained.

## Cold medians

All timings in milliseconds. Positive paired dense-minus-reuse values favor reuse.

| Workload | Dense | Reuse | Median paired dense minus reuse |
| --- | ---: | ---: | ---: |
| Single version | 15.8565 | 18.0430 | -3.0961 |
| Three versions | 49.2278 | 54.1520 | -5.2957 |

The original baseline single-version median was 20.3957 ms. These historical measurements were separate runs, not an alternating before/after experiment. Removing unnecessary dense work erased the earlier reported 17.76% reuse advantage against the original baseline.

## Persistent-cache medians

Initial construction is separate from updates.

| Version | Fresh dense | Cache | Median paired dense minus cache |
| --- | ---: | ---: | ---: |
| Initial original | 16.5115 | 18.9055 | -2.5880 |
| Insertion update | 16.4378 | 16.7847 | -0.5283 |
| Replacement update | 16.3237 | 16.4096 | 0.0347 |

Earlier update improvements of 6–9% did not hold in this pass. This stage did not establish a reliable update advantage.

Pinned-buffer/asynchronous-upload median paired savings were 0.8143 ms for single-version reuse and 0.4353 ms for three-version reuse. Pinning and required staging-buffer reuse synchronization are included. Pinning remains opt-in; these small differences need confirmation.

Every measured full output matched bitwise and all bytes reconstructed exactly. The local report records 46 passing tests, including cache configuration invalidation. Complete elapsed times include planning, packing, transfers, kernel execution, gathering, all host outputs, byte reconstruction and integrity checks. The FFT is cheap enough that CPU lookup and output materialization dominate.

Reproduce with the corresponding **local modified implementation**, not the older checked-in runner:

```powershell
python scripts/benchmark_gpu.py data/corpora --repeats 21 --output results/lossless/gpu_optimized.json
```
