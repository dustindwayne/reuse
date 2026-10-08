> Historical chunk-cache update stage. The update-only timings below do not establish complete lifecycle savings. Use the [current lifecycle findings](GPU_LIFECYCLE.md), [frozen repeatability report](GPU_LIFECYCLE_REPEATABILITY.md) and [GPU guide](GPU_COMPUTE.md) for the integrated implementation and current conclusions. One edit loses, two edits is unreliable, and four/eight edits are favorable across the three recorded sessions; no universal threshold is claimed.

# Chunk lookup and complete elapsed time

Hardware: RTX 5070 Ti, Torch 2.11.0+cu128, CUDA 12.8. Inputs: all 25 pinned
Canterbury/Calgary files, 5,952,406 original bytes. Kernel remains float32 rFFT
at width 2048 and batch cap 1024. No additional FFT passes were added.

## First: profile the existing block lookup path

21 alternating paired runs after warmup, single original version:

| Method | Planning and lookup median | Complete elapsed median |
| --- | ---: | ---: |
| Optimized dense | 6.2832 ms | 15.4448 ms |
| Block reuse | 9.4516 ms | 18.4683 ms |

Median paired complete difference (dense minus reuse): -2.5275 ms. The
3.1684 ms difference between planning medians explains most of the overall
regression. This timing covers planning and lookup together, including boundaries
and original file hashes; it does not isolate block lookup from those steps.
Raw results: `results/lossless/gpu_block_profile.json`.

## Chunk cache implementation

The public GPUCache now uses ChunkBlockPlan: scan the same content boundaries,
hash each chunk once, confirm exact byte equality on a hit, and reuse the entire
ordered range of GPU outputs. Split only new chunks with the same block offsets
and tail lengths as the dense plan. Keep SHA256 hashes for every complete file,
reconstruct from ordered chunks, and verify full size and hash on every call.
Cache identity or width changes invalidate byte lookup and GPU outputs together.
Legacy BlockGPUCache remains available for diagnostic comparisons.

## Cold construction and updates

Fresh cache for each paired sequence; one warmup sequence followed by 21 measured
sequences. Dense/cache order alternates. Cold timing includes cache construction.
Updates retain the original cache, then the insertion cache. Dense processes the
same full current file set independently each time.

| Phase | Dense median | Chunk cache median | Median paired saving | Median time reduction |
| --- | ---: | ---: | ---: | ---: |
| Cold original | 15.2087 ms | 18.9487 ms | -3.5042 ms | -24.59% |
| Insertion update | 15.9170 ms | 14.7189 ms | 0.9196 ms | 7.53% |
| Replacement update | 15.8411 ms | 13.6972 ms | 1.9873 ms | 13.53% |

Ratio-of-medians percentages and median paired differences are distinct statistics.
The earlier preliminary pass showed insertion close to a tie and replacement
faster, so the smaller insertion advantage needs cautious interpretation.

| Phase | Cache planning incl. lookup | Chunk lookup alone | New FFT blocks / complete output blocks |
| --- | ---: | ---: | ---: |
| Cold original | 8.3221 ms | 1.7654 ms | 3164 / 3502 |
| Insertion | 8.1405 ms | 2.4170 ms | 763 / 3503 |
| Replacement | 8.0283 ms | 2.5594 ms | 216 / 3503 |

Complete elapsed timing includes planning, SHA256 hashing, lookup/equality,
packing, transfers, FFT, cache append/gather, every host output, full byte
reconstruction and integrity checks. Source disk reads and independent dense/cache
output comparisons are outside the timer. GPU synchronization brackets timing.
Full outputs matched bitwise and reconstructed bytes matched exactly every run.
All 49 unit tests pass, including chunk collision rejection, original boundaries,
padding/order, reconstruction integrity and persistent configuration invalidation.

Cache retains immutable input buffers, copied chunk definitions and GPU outputs;
capacity eviction is not implemented. These results concern sequential constructed
edits and block-local FFT, with no energy measurement. Cold reuse remains slower.

Reproduce from the repository root:

```powershell
python scripts/benchmark_chunk_gpu.py data/corpora --profile-only
python scripts/benchmark_chunk_gpu.py data/corpora
```

Full measurements: `results/lossless/gpu_chunk_cache.json`.
