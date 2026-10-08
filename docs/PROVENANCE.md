# Evidence and attribution

Created by Dustin with AI implementation assistance.

Performance claims refer to committed code, pinned input manifests, raw measurements, and documented timing boundaries. The [current results](../RESULTS.md), [GPU lifecycle report](GPU_LIFECYCLE.md), and [repeatability report](GPU_LIFECYCLE_REPEATABILITY.md) describe the measured workloads and limits.

Input identities and hashes are recorded in [corpus sources](../results/lossless/corpus_sources.json) and [model-file sources](../results/lossless/fold_model_sources.json). The [recorded GPU sessions](../results/lossless/gpu_lifecycle_frozen_sessions/) contain measurements, logs, metadata, and the original source-hash manifest. [Validation](VALIDATION.md) distinguishes available checks from measurements requiring CUDA.

Earlier code and measured negative findings remain in the [experiment trail](EXPERIMENT_TRAIL.md) and [language-model archive](../archive/language-model/README.md). Those results retain their original scope.
