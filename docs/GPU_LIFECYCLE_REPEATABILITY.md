# Frozen lifecycle experiment: three independent benchmark sessions

The implementation and all benchmark dependency sources were frozen by SHA256.
Three fresh Python processes ran the identical lifecycle command sequentially.
Before and after each session, every frozen source hash was verified unchanged.
No tuning occurred. The original lifecycle result was preserved and restored;
new session JSON, logs, timestamps and the freeze manifest are saved under
`results/lossless/gpu_lifecycle_frozen_sessions/`.

Each session retains the same one warmup pair and 21 alternating complete
lifecycles for cold construction plus 1, 2, 4 and 8 cumulative constructed edits.
GPU: RTX 5070 Ti; Torch 2.11.0+cu128; CUDA 12.8; width 2048; batch cap 1024.
The input edits, kernel, precision, padding, output order, timing boundaries,
cache teardown and exactness checks are unchanged.

Positive paired saving means cache finishes faster. Totals are directly measured
complete lifecycles; phase medians are never summed.

| Session | Edits after cold construction | Dense median lifecycle | Cache median lifecycle | Median paired saving | Cache wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 32.6213 ms | 34.6489 ms | -2.1901 ms | 1 / 21 |
| 1 | 2 | 48.8761 ms | 47.7493 ms | 1.4325 ms | 14 / 21 |
| 1 | 4 | 82.7855 ms | 79.9958 ms | 2.5173 ms | 14 / 21 |
| 1 | 8 | 147.2796 ms | 139.4761 ms | 6.6586 ms | 20 / 21 |
| 2 | 1 | 33.9635 ms | 37.2252 ms | -2.7931 ms | 2 / 21 |
| 2 | 2 | 49.7365 ms | 49.0145 ms | -0.3735 ms | 9 / 21 |
| 2 | 4 | 83.3930 ms | 82.8291 ms | 1.7285 ms | 14 / 21 |
| 2 | 8 | 149.0526 ms | 140.6361 ms | 7.9156 ms | 20 / 21 |
| 3 | 1 | 34.5671 ms | 35.9317 ms | -0.6030 ms | 8 / 21 |
| 3 | 2 | 51.9948 ms | 51.2801 ms | 0.0063 ms | 12 / 21 |
| 3 | 4 | 85.5402 ms | 83.0873 ms | 2.1370 ms | 18 / 21 |
| 3 | 8 | 150.2181 ms | 144.2128 ms | 9.4338 ms | 19 / 21 |

The small two-edit advantage does not reliably hold: one session loses on the
median paired comparison, and another is effectively tied (0.0063 ms saving).
Four edits is the earliest tested point with positive median paired savings and
majority cache wins in every session. Eight edits gives the strongest repeated
advantage. One edit loses in all three sessions.

Differences between the separate dense/cache medians are not the same statistic
as the median paired difference; session 2 with two edits illustrates why pairing
matters. No guaranteed edit-count threshold or statistical confidence interval is
claimed. These sequential constructed edits modify every corpus file and do not
represent recorded editing histories.

Across 252 measured lifecycle pairs (84 per session), every full output from
every version matched bitwise and every reconstructed input matched exactly.
Timing includes construction, preparation and chunk lookup/equality, file hashes,
packing, transfers, unchanged FFT, output/cache maintenance, full output return,
reconstruction/integrity checks and teardown. Independent output comparisons,
source reads, synthetic edit generation and pre-pair GC remain outside timing,
exactly as in the frozen benchmark. Each file's original SHA256 is verified during
input loading, and reconstructed-file integrity is checked on every call.

Commands used in separate processes:

```powershell
python scripts/repeat_gpu_lifecycle.py 1
python scripts/repeat_gpu_lifecycle.py 2
python scripts/repeat_gpu_lifecycle.py 3
```

The wrapper refuses to overwrite existing session files, launches the unchanged
benchmark in a new process, verifies freeze hashes and preserves output records.
