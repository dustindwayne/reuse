# Shared Data Representation

## Reduced-compute path

The [fast shared store](docs/FAST_COMPUTE.md) processes complete files and reconstructs every byte. It uses bounded content boundary searches, stores repeated chunks once, and encodes each new chunk once. On all 25 Canterbury and Calgary files, complete encode-plus-restore CPU time fell 42.7% against a matched whole-file cached store. With two constructed edited versions, CPU time fell 68.3%. The single-version archive is 17.2% larger; these are CPU measurements for this workload, not energy measurements or a universal speedup.

```bash
python scripts/fast_store.py pack versions.fst snapshots/original snapshots/edited
python scripts/fast_store.py unpack versions.fst restored
python scripts/benchmark_fast.py /path/to/corpora
```

## GPU computation reuse

The [CUDA chunk cache](docs/GPU_COMPUTE.md) hashes and verifies each content chunk once, computes new chunks' block outputs, and retains those outputs across constructed edits. It preserves the dense path's block boundaries, padding and output order, reconstructs every byte and verifies file SHA256 hashes. The operation tested is float32 block-local rFFT, width 2048, batch cap 1024.

The primary result is **complete lifecycle elapsed time**, including cold construction, all updates, every full output, integrity checks and cache teardown. Three [frozen benchmark sessions](docs/GPU_LIFECYCLE_REPEATABILITY.md) on an RTX 5070 Ti used 21 alternating pairs per workload:

| Constructed edits after cold start | Median paired savings across the three sessions | Cache wins per session |
| ---: | --- | --- |
| 1 | -2.19, -2.79, -0.60 ms | 1/21, 2/21, 8/21 |
| 2 | +1.43, -0.37, +0.006 ms | 14/21, 9/21, 12/21 |
| 4 | +2.52, +1.73, +2.14 ms | 14/21, 14/21, 18/21 |
| 8 | +6.66, +7.92, +9.43 ms | 20/21, 20/21, 19/21 |

The two-edit margin is unstable; four and eight edits improve time in all three sessions. These are cumulative constructed edits of all 25 pinned Canterbury/Calgary files, not observed editing histories. Every output matched bitwise and every byte reconstructed exactly. The cache is currently unbounded. These measurements establish neither neural-inference nor energy savings; cold caching remains slower. Earlier experiments and negative results remain in the [experiment history](docs/GPU_COMPUTE.md#experiment-history).

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

## About this work

I build these experiments because I want to understand how things work and whether they can work better. Curiosity and learning are the reason I keep going: ask a question, build something, test it, and change what the evidence says needs changing.

The [integrated data container](docs/CONTAINER.md) stores shared chunks once across files and chooses among raw storage, dynamic windows, repeated-value extraction, shared grammar, grid offsets and zlib by actual encoded size. Structural transforms can also feed zlib; [composition measurements](docs/COMPOSED.md) record the results. It preserves exact file contents. Run `python scripts/benchmark_container.py`.

Current direction: a **lossless shared representation** that stores distinct pieces once and preserves exact reconstruction. The new [lossless foundation](docs/LOSSLESS.md) implements shared byte and sequence definitions, a binary file format, exact decoding and measured storage comparisons.


```bash
python scripts/benchmark_lossless.py
python scripts/lossless.py pack data/train.txt train.lsg
python scripts/lossless.py unpack train.lsg restored.txt
```

## Grid average experiment

The [grid experiment](docs/GRID.md) groups bytes into rectangular blocks and stores an integer average plus exact bit-packed offsets, choosing raw blocks when smaller. It saves space on clustered values and reconstructs exactly; random bytes and text do not benefit in these tests. Run `python scripts/benchmark_grid.py`.

## Dynamic windows and repeated values

The [adaptive experiment](docs/ADAPTIVE.md) selects window sizes by actual encoded cost and can extract a repeated value into a bitmap before compressing the remainder. It reconstructs exactly and keeps whole objects raw when compression loses. Run `python scripts/benchmark_adaptive.py`; measured comparisons isolate splitting and extraction.

## Real-corpus evaluation

[Complete Canterbury and Calgary tests](docs/CORPUS_EVALUATION.md): 25 full files, 5.95 MB, exact reconstruction throughout. Whole-file transforms followed by zlib reduce payload totals by about 2.7% against direct zlib. The earlier Python candidate search is much slower than direct compression; the reduced-compute path above replaces that search. Per-file baselines, matched controls, chunk sensitivity and reproduction scripts are included.

## Pattern matching before encoding

The [pattern-first path](docs/PATTERN_FIRST.md) finds exact reusable ranges before establishing literal boundaries, including shifted data and cross-file matches. It stores backward references and applies the remaining framework to new literal data. Shifted-copy tests improve substantially; the current greedy matcher slightly enlarges the real-corpus archives against the same pipeline without matching. Run `python scripts/container.py pack --patterns example.ptf data/train.txt data/validation.txt`.

## Recovered lane transforms and complete-path selection

The [recovered-framework integration](docs/RECOVERED_INTEGRATION.md) adds reversible lane folding and a 3/1/3 permutation before zlib. It also compares pooled, pattern-first and file-aware complete archives. Calgary’s archive improves by 3.94% against the prior complete path; Canterbury retains its smaller existing encoding. Run `python scripts/container.py pack --auto example.udf data/train.txt data/validation.txt`.

## Earlier lossy experiment

The sections below describe the earlier implementation and its historical measurements. It did not implement the lossless shared representation or residual-only neural prediction.

## Research question

Can predictable linguistic structure be handled deterministically so that neural computation is concentrated on residual uncertainty?

A compact encoder represents recurring orthographic features. A train-only context trie predicts structural states. The mismatch on an already observed token supplies a residual and surprise signal, which selects neural depth for the next-token prediction.

## Architecture

The system combines a fixed-context neural LM with a deterministic prior. The adaptive configuration uses the observed token's surprise to select zero, one or two optional blocks. See [architecture](ARCHITECTURE.md) for the causal indexing, equations and parameter differences.

## Current result

The prototype skips optional neural block work while keeping similar loss on a synthetic grammar smoke corpus. **It does not demonstrate an end-to-end speedup:** structural feature extraction dominates this small model. Three-seed comparisons and randomized controls are in [results](RESULTS.md); full metrics and per-example gate components are committed.

## Reproduce

Python 3.10+ on Linux; no model downloads, account, API key or GPU required. Process memory measurement currently uses Unix `resource`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/prepare_data.py
python scripts/benchmark.py --config configs/baseline.yaml
python scripts/benchmark.py --config configs/structural.yaml
python scripts/benchmark.py --config configs/residual.yaml
python scripts/predict.py --text "the frog"
python scripts/run_ablations.py --seeds 7,11,23
```

To use a local natural-text corpus, run `python scripts/prepare_data.py --input corpus.txt`. Each nonempty line is a document; identical lines are deduplicated before the seeded split. This replaces smoke data, so record its license, identity and resulting hashes before sharing. Configuration paths are relative to the repository root.

Benchmarks save JSON metrics, a feature/gate CSV for structural systems and local checkpoints under ignored `results/checkpoints`. `predict.py` prints top next-token probabilities from a trained checkpoint; it verifies corpus hashes for the deterministic prior. Results include the actual configuration, seed, dataset hashes, Git commit, environment, timing repeats and parameter count.

## Evidence and limits

- [Measured results and ablations](RESULTS.md)
- [Mechanism and reasoning](ARCHITECTURE.md)
- [Provenance and attribution](docs/PROVENANCE.md)
- [Experiment design](docs/EXPERIMENTS.md)
- [Failures and corrections](docs/FAILURE_LOG.md)
- [Resume description](docs/RESUME.md)
- [Validation and remaining checks](docs/VALIDATION.md)

This is a small fixed-context feed-forward model and designed grammar dataset. It does not establish Transformer scaling, novel-vocabulary generalization, semantic reasoning, full-model paging or production savings. Structural configurations add feature-projection parameters. Skipped block work is not total FLOPs or wall-time reduction. Three seeds and small loss differences do not establish statistical superiority. The confidence/surprise formula and thresholds are experimental choices, not calibrated measures of difficulty.

MIT licensed. Development commits preserve the baseline-before-integration sequence and the timing correction. Historical provenance is distinct from new code implemented with AI assistance.
