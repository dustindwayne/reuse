# GPU lifecycle elapsed time: frozen repeatability results

**Current evidence:** supplied desktop measurements show repeatable complete lifecycle savings with four and eight constructed edits. One edit loses in all sessions; two edits is unreliable. Four edits is the earliest tested consistently favorable point, not a guaranteed threshold. Complete elapsed time is the primary metric.

Provenance: reports supplied on 2026-10-08 from the desktop Work task. This session has not inspected the local optimized source or raw JSON. The optimized implementation, raw sessions, logs and freeze manifest remain local and uncommitted. This document records those reports; it does not claim the newer implementation is present in this checkout. The [checked-in GPU prototype](archive/GPU_PROTOTYPE.md) predates these optimizations.

## Frozen experiment

RTX 5070 Ti, PyTorch 2.11.0+cu128, CUDA 12.8. All 25 pinned Canterbury/Calgary files, 5,952,406 original bytes. One float32 block-local rFFT per required zero-padded block, width 2048 and batch cap 1024. No artificial extra FFT work was introduced.

SHA-256 hashes froze the implementation and benchmark dependency sources. Three separate fresh Python processes ran sequentially, with every frozen hash verified unchanged before and after each session. No tuning occurred. Each edit-count experiment used one warmup pair and 21 alternating paired lifecycles, each starting a fresh cache.

Odd cumulative steps insert a constructed record at one-third of every current file. Even steps replace 17 bytes at two-thirds with an edit record. Every step processes every complete current file. These are constructed edits of real corpus files, not recorded editing histories.

Each total is directly timed from cold construction through all updates and cache teardown. Positive median paired savings mean cache finished faster. The separate medians and median paired difference are distinct statistics.

| Session | Edits after cold start | Dense median ms | Cache median ms | Median paired saving ms | Cache wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 32.6213 | 34.6489 | -2.1901 | 1/21 |
| 1 | 2 | 48.8761 | 47.7493 | 1.4325 | 14/21 |
| 1 | 4 | 82.7855 | 79.9958 | 2.5173 | 14/21 |
| 1 | 8 | 147.2796 | 139.4761 | 6.6586 | 20/21 |
| 2 | 1 | 33.9635 | 37.2252 | -2.7931 | 2/21 |
| 2 | 2 | 49.7365 | 49.0145 | -0.3735 | 9/21 |
| 2 | 4 | 83.3930 | 82.8291 | 1.7285 | 14/21 |
| 2 | 8 | 149.0526 | 140.6361 | 7.9156 | 20/21 |
| 3 | 1 | 34.5671 | 35.9317 | -0.6030 | 8/21 |
| 3 | 2 | 51.9948 | 51.2801 | 0.0063 | 12/21 |
| 3 | 4 | 85.5402 | 83.0873 | 2.1370 | 18/21 |
| 3 | 8 | 150.2181 | 144.2128 | 9.4338 | 19/21 |

The initial lifecycle report's two-edit startup recovery conclusion is superseded. Session 2 loses on the paired two-edit comparison, and session 3 is effectively tied. Four edits has positive median paired savings and majority wins in every session. Eight edits gives the strongest repeated advantage. No confidence interval or universal edit-count threshold is claimed.

## Timing and exactness

The complete timer includes construction, preparation, chunk lookup and exact equality, file hashes, packing, H2D transfers, unchanged FFT, GPU output/cache maintenance, full D2H outputs, reconstruction, integrity checks, additional original-byte comparison and teardown. Synchronization brackets the lifecycle and each call. Full host outputs from every version are retained for independent comparison.

Source disk reads, synthetic edit generation, pre-pair garbage collection and independent dense/cache full-output comparisons are outside timing. Each source file's SHA-256 is verified during loading; each reconstructed file's integrity is checked on every call.

All **252 measured lifecycle pairs / 1,197 full version comparisons** passed byte-exact reconstruction and bitwise full-output equality. The supplied local report records 49 passing unit tests, including batches bisecting chunk groups, dirty staging buffers, exact tail padding, collision rejection, integrity verification and configuration invalidation. The checked-in older prototype has 45 tests; these counts refer to different implementations.

## Local implementation and reproduction gap

The local cache hashes each content chunk once, confirms byte equality, and reuses an ordered range of GPU block outputs. Chunk references are built once and reused as tuples. Newly required input rows use contiguous NumPy copies per chunk; binary search locates the first intersecting chunk for a transfer batch. Padding, block offsets and output ordering match dense processing. Kernel identity or width changes invalidate lookup and GPU outputs together. Cache capacity is unbounded and input buffers must remain immutable.

Latest local sources include `src/structural_residual_lm/gpu_reuse.py`, `scripts/benchmark_gpu.py`, `scripts/benchmark_chunk_gpu.py`, `scripts/benchmark_gpu_lifecycle.py`, `scripts/repeat_gpu_lifecycle.py`, and `tests/test_gpu_reuse.py`. Their updated contents have not been supplied to this checkout. Do not reconstruct or claim them from summary measurements.

On the desktop, the unchanged frozen experiment was run in separate processes:

```powershell
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/repeat_gpu_lifecycle.py 1
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/repeat_gpu_lifecycle.py 2
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/repeat_gpu_lifecycle.py 3
```

These commands require the **local newer implementation**; the repeat wrapper is not yet checked in. It refuses overwrites, verifies the freeze hashes and preserves output records. Raw JSON, logs, timestamps and the source-freeze manifest are under local `results/lossless/gpu_lifecycle_frozen_sessions/`. Earlier raw measurements remain in `gpu_existing_lifecycle.json` and `gpu_lifecycle.json`. No raw files were fabricated from report tables.

## Scope and next checks

Supported scope: complete elapsed-time improvement for repeated constructed edits, on this hardware, using block-local FFT and exact reuse. This is not evidence of general neural inference acceleration or electricity savings. Unchanged-file, low-reuse and bounded-capacity behavior remain to be evaluated. The repo stays private pending source/raw-data integration and review of the scoped claim.

[Experiment trail](EXPERIMENT_TRAIL.md) preserves unsuccessful paths and corrected conclusions.
