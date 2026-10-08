> Historical record. Superseded by the [frozen lifecycle findings](../GPU_LIFECYCLE.md). Older conclusions remain here to preserve the experiment trail.

# Chunk lookup and complete elapsed time

Provenance: supplied local-run report, received 2026-10-08. The report supersedes the earlier [block-cache update conclusions](GPU_LOCAL_OPTIMIZATION.md). This session has not inspected the local implementation or raw run JSON. Local changes and raw timings remain uncommitted on the desktop; this commit records the supplied results only.

Hardware: RTX 5070 Ti, Torch 2.11.0+cu128, CUDA 12.8. All 25 pinned Canterbury/Calgary files, 5,952,406 original bytes. Float32 block-local rFFT, width 2048, batch cap 1024; no extra FFT passes were added.

## Profile of block lookup

21 alternating paired runs after warmup, single original version. Milliseconds:

| Method | Planning and lookup median | Complete elapsed median |
| --- | ---: | ---: |
| Optimized dense | 6.2832 | 15.4448 |
| Block reuse | 9.4516 | 18.4683 |

Median paired complete difference, dense minus reuse: -2.5275 ms. The 3.1684 ms difference between planning medians explains most of the regression. Planning includes boundaries and file hashes; it does not isolate block lookup. Local raw results: `results/lossless/gpu_block_profile.json` (not included here).

## Local chunk-cache implementation

The locally modified GPUCache uses ChunkBlockPlan: scan the same content boundaries, hash each chunk once, confirm exact byte equality, and reuse the entire ordered range of GPU outputs. Split only new chunks using the same block offsets and tail lengths as dense processing. Retain SHA-256 for complete files, reconstruct ordered chunks, and verify full size and hash on each call. Kernel identity or width changes invalidate byte lookup and GPU outputs together. Legacy BlockGPUCache remains for diagnostics.

These classes are described from the supplied report; they are **not yet integrated into the checked-in source**.

## Cold construction and updates

Fresh cache for each paired sequence; one warmup sequence and 21 measured sequences. Dense/cache order alternates. Cold time includes cache construction. Updates retain preceding cache state. Dense independently processes the same full current files. Positive paired savings favor cache.

| Phase | Dense median ms | Chunk cache median ms | Median paired saving ms | Ratio-of-medians time reduction |
| --- | ---: | ---: | ---: | ---: |
| Cold original | 15.2087 | 18.9487 | -3.5042 | -24.59% |
| Insertion update | 15.9170 | 14.7189 | 0.9196 | 7.53% |
| Replacement update | 15.8411 | 13.6972 | 1.9873 | 13.53% |

Ratio-of-medians percentages and median paired differences are distinct statistics. An earlier preliminary pass had insertion near a tie; its smaller advantage needs cautious interpretation.

| Phase | Cache planning incl. lookup ms | Chunk lookup alone ms | New FFT blocks / complete output blocks |
| --- | ---: | ---: | ---: |
| Cold original | 8.3221 | 1.7654 | 3164 / 3502 |
| Insertion | 8.1405 | 2.4170 | 763 / 3503 |
| Replacement | 8.0283 | 2.5594 | 216 / 3503 |

The complete timer includes planning, SHA-256 hashing, lookup/equality, packing, transfers, FFT, cache append/gather, every host output, full byte reconstruction and integrity checks. Source disk reads and independent dense/cache output comparisons are outside timing. GPU synchronization brackets timing.

All full computed outputs matched bitwise and reconstructed bytes matched exactly on every measured run. The local report records 49 passing tests, including chunk collision rejection, boundaries, padding/order, reconstruction integrity and persistent configuration invalidation. The cache retains immutable input buffers, copied chunk definitions and GPU outputs; eviction is not implemented.

## What is established and what remains open

The supplied measurements show faster complete processing for these updates, at a slower initial construction cost. They concern sequential constructed edits of real corpus files and block-local FFT. They do not establish a complete lifecycle advantage, general neural inference speedup or energy savings.

As a diagnostic only, summing phase medians gives 46.9668 ms dense and 47.3648 ms cached. This is **not** the median of sequence totals and cannot establish a lifecycle result. The next check is each raw sequence's cold-plus-updates total, followed by measured amortization over more updates. Complete elapsed time remains the objective; fewer computed blocks or lower VRAM alone are insufficient. Public readiness remains on hold.

Reproduce from the repository root with the **local modified implementation**; this script is not yet checked in:

```powershell
python scripts/benchmark_chunk_gpu.py data/corpora --profile-only
python scripts/benchmark_chunk_gpu.py data/corpora
```

Local raw measurements: `results/lossless/gpu_chunk_cache.json` (not included in this documentation commit).
