# Shared Data Representation

Reduce complete processing time by storing repeated data once and reusing computation where its inputs and operation match. Preserve every input byte and every required output. Smaller storage, fewer computed blocks and lower VRAM are separate measurements from elapsed-time improvement.

## About this work

I build these experiments because I want to understand how things work and whether they can work better. Curiosity and learning are the reason I keep going: ask a question, build something, test it, and change what the evidence says needs changing.

## Current GPU evidence

The committed [frozen local lifecycle measurements](docs/GPU_LIFECYCLE.md) on an RTX 5070 Ti show positive median paired **complete elapsed-time savings** after four and eight constructed edits across three independent benchmark sessions. At eight edits, paired median savings range from 6.66 to 9.43 ms; cache wins 19–20 of 21 pairs per session. One edit loses in all sessions; two edits is unreliable.

All 252 measured lifecycle pairs preserved byte-exact reconstruction and bitwise-equal full outputs. Timing includes cold construction, all processing and output return, verification, and teardown. The workload is block-local FFT over full real corpus files with constructed edits; it does not establish general neural inference or electricity savings.

**Integration status:** the tested 49-test implementation, complete raw measurements, three frozen session logs and source-freeze manifest are integrated. The repository remains private. See the [current report](docs/GPU_LIFECYCLE.md) and [implementation/reproduction guide](docs/GPU_COMPUTE.md).

## GPU verification and reproduction

Use an existing CUDA-enabled Python environment (tested: Python 3.13.13, Torch 2.11.0+cu128, NumPy 2.5.3). Torch is optional for CPU-only tests and is not installed by the project. From the repository root:

```powershell
conda activate torch5070
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/verify_gpu_integration.py
```

The verification command checks committed result consistency and the freeze manifest; with CUDA available it checks exact full outputs and reconstruction on a small fixture, without adding benchmark measurements. To reproduce the existing full-corpus experiment:

```powershell
python scripts/download_corpora.py data/corpora
python scripts/benchmark_gpu_lifecycle.py data/corpora --repeats 21 --width 2048 --batch-blocks 1024
```

That writes `results/lossless/gpu_lifecycle.json`; the original recorded lifecycle also remains archived under `results/lossless/gpu_lifecycle_frozen_sessions/original_lifecycle.json`. For three independent new sessions, use a **new** archive directory so the recorded sessions are never overwritten:

```powershell
python scripts/repeat_gpu_lifecycle.py 1 --archive-dir results/reproductions/new-lifecycle-sessions
python scripts/repeat_gpu_lifecycle.py 2 --archive-dir results/reproductions/new-lifecycle-sessions
python scripts/repeat_gpu_lifecycle.py 3 --archive-dir results/reproductions/new-lifecycle-sessions
```

The Windows helper `scripts/run_gpu.ps1` uses the lifecycle benchmark and defaults to 21 pairs. The pinned inputs require a public download; verification never substitutes CPU timing for CUDA results.

## Runnable CPU path

The [fast shared store](docs/FAST_COMPUTE.md) finds bounded content boundaries, stores repeated chunks once and encodes each new chunk once. Complete encode-plus-restore **CPU time** fell 42.7% on all 25 original Canterbury/Calgary files and 68.3% over two constructed edited versions against a matched whole-file cached store. The single-version archive is 17.2% larger. These CPU measurements are distinct from the GPU elapsed-time experiments.

```bash
python scripts/fast_store.py pack versions.fst snapshots/original snapshots/edited
python scripts/fast_store.py unpack versions.fst restored
python scripts/benchmark_fast.py /path/to/corpora
```

## Evidence trail

[Experiment trail](docs/EXPERIMENT_TRAIL.md) links earlier representations, failed approaches, baseline corrections and historical language-model work. Earlier reports are archived rather than presented as current findings. [Validation](docs/VALIDATION.md) separates checked-in tests from reported local tests.

MIT licensed. Historical ideas belong to Dustin; implementation assistance and source limits are documented in [provenance](docs/PROVENANCE.md).
