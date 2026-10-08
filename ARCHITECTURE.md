# Current architecture

The active framework preserves complete input bytes and required outputs while trying to reduce complete elapsed processing time.

The CPU shared store finds content boundaries, stores equal chunks once, and restores ordered files from references. The GPU path uses the same boundaries, hashes each chunk, confirms exact equality on hits, and reuses its ordered block outputs. New chunks alone are packed and processed. Full outputs return to the host, inputs reconstruct exactly, and file size and SHA256 integrity are checked.

Cache identity and block width invalidate lookup and outputs together. Inputs must remain immutable, and callers must change the identity for each operation or configuration change. The current cache has no capacity eviction.

The measured GPU operation is block-local float32 rFFT, not a neural model. Complete lifecycle timing includes startup, preparation, transfers, full output return, reconstruction, verification, and teardown. See [GPU implementation](docs/GPU_COMPUTE.md), [lifecycle findings](docs/GPU_LIFECYCLE.md), and [CPU shared store](docs/FAST_COMPUTE.md).

The [earlier language-model architecture](archive/language-model/ARCHITECTURE.md) belongs to a separate archived experiment. The existing `structural_residual_lm` Python namespace remains for compatibility.
