# GPU chunk reuse and complete lifecycle timing

The tested implementation is `GPUCache` / `ChunkGPUCache` in
`src/structural_residual_lm/gpu_reuse.py`. Each content chunk is hashed once and
confirmed byte-equal before reusing its entire ordered group of GPU outputs.
Only new chunks are split and evaluated. Kernel identity or width changes clear
the cache; callers must change identity for every operation/configuration change
and keep input buffers immutable. Cache capacity is currently unbounded.

The dense baseline bypasses unused block hashes and identity gathering. Both
paths preserve content boundaries, block offsets, zero padding and output order,
return every full output and reconstruct every file with size/SHA256 checks.
The operation is float32 block-local rFFT, width 2048, batch cap 1024.

## Latest measurements

Complete cold-plus-edit lifecycle elapsed time is primary, including cache teardown.
Three fresh benchmark sessions on RTX 5070 Ti / Torch 2.11.0+cu128 / CUDA 12.8
each used 21 alternating pairs per edit count. One edit loses in all sessions;
two-edit gains are unstable. Four and eight constructed edits improve paired
median time and win a majority of pairs in every session. See the
[complete repeatability report](GPU_LIFECYCLE_REPEATABILITY.md).
All 252 lifecycle pairs / 1,197 version comparisons matched bitwise and
reconstructed byte-exactly. These cumulative edits modify all 25 pinned corpus
files; they are constructed tests, not recorded editing histories.
Cold reuse remains slower; these are neither neural-inference nor energy savings.

## Reproduction and verification

Use an existing CUDA-enabled environment; helpers do not install Torch or drivers.
Torch is optional for CPU unit tests.

```powershell
conda activate torch5070
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/verify_gpu_integration.py
python scripts/download_corpora.py data/corpora
python scripts/benchmark_gpu_lifecycle.py data/corpora --repeats 21 --width 2048 --batch-blocks 1024
```

The lifecycle command writes `results/lossless/gpu_lifecycle.json`; the original
also remains archived in
`results/lossless/gpu_lifecycle_frozen_sessions/original_lifecycle.json`.
For three independent repetitions, select a new archive directory:

```powershell
python scripts/repeat_gpu_lifecycle.py 1 --archive-dir results/reproductions/new-lifecycle-sessions
python scripts/repeat_gpu_lifecycle.py 2 --archive-dir results/reproductions/new-lifecycle-sessions
python scripts/repeat_gpu_lifecycle.py 3 --archive-dir results/reproductions/new-lifecycle-sessions
```

Existing session files are never overwritten. `scripts/run_gpu.ps1` now invokes
the lifecycle benchmark with 21 pairs by default. Successful completion means
exact measurement succeeded; inspect paired savings and wins to judge performance.
The historical requirement that cold reuse also beat dense was not met.
Committing scoped evidence does not authorize making the repository public.

`verify_gpu_integration.py` checks frozen source hashes and committed statistics,
then verifies CUDA outputs and reconstruction on nine versions of a small fixture,
without recording new timing results. Pass `--corpora data/corpora` for all pinned
files. CUDA unavailability is explicitly reported.

## Experiment history

- [Original pre-hardware handoff](history/GPU_COMPUTE_PENDING.md) and
  `results/lossless/history/gpu_compute_pending.json`: initial blocked result.
- `results/lossless/gpu_compute.json`: first local 21-repeat block-reuse result.
- [Preparation/gather optimization](GPU_LOCAL_OPTIMIZATION.md) and
  `gpu_optimized.json`: earlier implementation stage.
- [Chunk lookup](GPU_CHUNK_LOOKUP.md): `gpu_block_profile.json`,
  `gpu_chunk_cache.json`.
- [Complete lifecycles](GPU_LIFECYCLE.md): `gpu_existing_lifecycle.json`,
  `gpu_lifecycle.json`.
- [Frozen repeatability](GPU_LIFECYCLE_REPEATABILITY.md): three session JSONs,
  logs, metadata, original lifecycle and freeze manifest under
  `results/lossless/gpu_lifecycle_frozen_sessions/`.

Raw JSON names above are relative to `results/lossless/`. Historical commands
and conclusions describe their earlier stages; use current commands above for
integrated reproduction. The freeze manifest covers benchmark sources, not
later README/helper/verification edits. Git attributes pin source line endings
to retain the recorded hashes on clean Windows and Linux checkouts.

Source fixes after the recorded sessions are not GPU-timed. The verifier resolves original measured sources through `archive/path-map.json`; this is historical evidence verification, not proof that current source hashes match the measured implementation. Use a new archive directory for any future runs.
