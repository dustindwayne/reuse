# Complete GPU cache lifecycle elapsed time

Primary metric: complete elapsed time from cold construction through every edit,
including cache teardown. GPU: RTX 5070 Ti, Torch 2.11.0+cu128, CUDA 12.8.
The operation remains one float32 rFFT per required zero-padded block, width 2048,
batch cap 1024. No extra FFT work was introduced.

## Existing 21 paired sequences

For each matching sequence index in `gpu_chunk_cache.json`, sum cold + insertion
+ replacement for dense and cache separately, then calculate statistics across
those 21 sequence totals. Phase medians were not summed.

| Dense median total | Cache median total | Median paired saving | Cache wins |
| ---: | ---: | ---: | ---: |
| 46.9038 ms | 48.0330 ms | -0.3065 ms | 9 / 21 |

These totals use the existing individual call timers; they cannot retrospectively
include inter-call or final cache teardown costs absent from the original timers.
Full per-sequence totals: `results/lossless/gpu_existing_lifecycle.json`.

## Planning and construction changes

Cache chunk references are now built once and reused as a tuple on subsequent
hits. Newly required input rows are packed with contiguous NumPy copies per
chunk instead of a Python/NumPy copy for every block. Batch packing locates the
first intersecting chunk by binary search, preserves partial groups and exact
tail padding, and uses the existing reusable staging buffer. SHA256 file integrity
checks, chunk hashes and verified exact equality remain in place.

These implementation changes target preparation cost. This experiment compares
the changed cache to the optimized dense baseline; it does not isolate their
before/after speedup. Planning remains substantial: in the eight-edit experiment,
cold planning was 9.34 ms and update planning medians were 7.94–8.45 ms. Additional
planning overhead reduction is still needed; no planning speedup is claimed.

## New complete lifecycle measurement

All 25 original pinned Canterbury/Calgary files (5,952,406 bytes) are processed
first. Each edit modifies all 25 complete files and is cumulative. Odd-numbered
steps insert `\nconstructed lifecycle insertion N\n` at one-third of each current
file. Even-numbered steps replace 17 bytes at two-thirds with `edit-NN-record`.
These are constructed test edits, not recorded editing histories.

Each edit-count experiment uses one warmup pair followed by 21 measured pairs.
Every pair starts a fresh cache. Dense and cache run complete independent
lifecycles in alternating order. Each reported total is measured directly across
all calls and teardown; none is a sum of phase medians.

| Constructed edits after cold start | Dense median lifecycle | Cache median lifecycle | Median paired saving | Cache wins |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33.5610 ms | 36.5212 ms | -2.6569 ms | 2 / 21 |
| 2 | 49.6199 ms | 48.8760 ms | 0.9376 ms | 14 / 21 |
| 4 | 84.1073 ms | 81.5887 ms | 2.4128 ms | 17 / 21 |
| 8 | 152.6047 ms | 147.6919 ms | 5.1617 ms | 18 / 21 |

Startup cost is recovered at the first tested point of two constructed edits,
judged by positive median paired savings and a majority of cache wins. This is
a workload-dependent measurement, not a guaranteed threshold. At eight edits,
the difference of median lifecycle totals is about 3.22%; median paired saving
is a distinct statistic. At two edits the margin and win rate remain modest.

Complete timing includes cache construction, planning and lookup, hashing and
exact equality, packing, H2D, unchanged FFT, GPU output append/gather, every full
D2H result, full reconstruction and SHA256 verification, additional original-byte
comparison, and cache teardown. Synchronization brackets the lifecycle and each
call. Disk reads, constructed edit generation, pre-pair garbage collection and
independent dense/cache full-output comparisons are outside the measured timer.
Full host outputs from every version are retained until independently compared.
Every measured version matched bitwise and reconstructed exactly.

All 49 tests pass. Verification includes transfer batches that bisect chunk groups,
reuse of dirty staging arrays, exact tail padding, collision rejection, integrity
checks and cache configuration invalidation. Cache capacity remains unbounded.

Run:

```powershell
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/benchmark_gpu_lifecycle.py --existing-only
& 'C:\Users\ddwan\anaconda3\envs\torch5070\python.exe' scripts/benchmark_gpu_lifecycle.py data/corpora --repeats 21
```

All lifecycle runs and secondary phase timings:
`results/lossless/gpu_lifecycle.json`.
