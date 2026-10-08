# Initial complete GPU lifecycle experiment

Historical supplied local report, 2026-10-08. Superseded by the [three-session frozen results](../GPU_LIFECYCLE.md). The initially favorable two-edit point did not reliably repeat.

On RTX 5070 Ti / Torch 2.11.0+cu128 / CUDA 12.8, width 2048, batch cap 1024, the existing 21 cold+insertion+replacement sequence timers gave dense median total 46.9038 ms, cache median 48.0330 ms, median paired saving -0.3065 ms and cache wins 9/21. These sums used actual per-sequence calls, not phase medians, but could not retroactively include inter-call and teardown costs absent from those timers. Local raw file: `results/lossless/gpu_existing_lifecycle.json`.

Chunk reference tuples and contiguous chunk packing were then introduced. Partial transfer groups were located by binary search while preserving padding and ordering. This compared the changed cache to dense; it did not isolate a before/after implementation effect. Planning remained substantial: cold 9.34 ms and update medians 7.94–8.45 ms in the eight-edit experiment.

A directly measured complete lifecycle included cold construction, all full edit calls and teardown. One warmup pair and 21 alternating measured pairs per edit count:

| Edits | Dense median ms | Cache median ms | Median paired saving ms | Cache wins |
| --- | ---: | ---: | ---: | ---: |
| 1 | 33.5610 | 36.5212 | -2.6569 | 2/21 |
| 2 | 49.6199 | 48.8760 | 0.9376 | 14/21 |
| 4 | 84.1073 | 81.5887 | 2.4128 | 17/21 |
| 8 | 152.6047 | 147.6919 | 5.1617 | 18/21 |

All 25 complete real corpus files were processed at each step. Cumulative edits alternated insertion at one-third and replacement at two-thirds. The FFT work was unchanged. Timers included planning, hashing/equality, transfers, computation, cache maintenance, all full outputs, byte reconstruction, integrity checks and teardown. Source reads, edit generation, pre-pair GC and independent output comparisons were excluded. Exact outputs and full reconstruction passed, with 49 local unit tests passing.

This first pass suggested startup recovery at two edits. The later frozen sessions showed that conclusion was not reliable. Local raw file: `results/lossless/gpu_lifecycle.json`; neither raw file has been supplied to this checkout.
