# Current results

Complete elapsed time is the primary metric. All input bytes and required outputs must be preserved.

Three independent RTX 5070 Ti lifecycle sessions show positive median paired savings and majority cache wins after four and eight constructed edits. One edit loses in every session; two edits is unreliable. At eight edits, paired median savings are 6.66–9.43 ms, with 19–20 wins out of 21 pairs per session. These are workload-specific results, not a guaranteed break-even threshold.

All 252 measured lifecycle pairs preserved byte-exact reconstruction and bitwise-equal full outputs. The operation is block-local float32 rFFT over 25 pinned real corpus files with constructed edits. No general neural-inference or electricity saving is established.

[Lifecycle report](docs/GPU_LIFECYCLE.md), [repeatability report](docs/GPU_LIFECYCLE_REPEATABILITY.md), and [raw sessions](results/lossless/gpu_lifecycle_frozen_sessions/) contain the measurements and limits.

The [CPU shared-store report](docs/FAST_COMPUTE.md) records a separate encode-plus-restore CPU-time improvement with a storage tradeoff. [Historical language-model results](archive/language-model/RESULTS.md) and the [experiment trail](docs/EXPERIMENT_TRAIL.md) preserve earlier work and negative findings.
