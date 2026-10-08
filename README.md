# Shared Data Representation

## Reduced-compute path

The [fast shared store](docs/FAST_COMPUTE.md) processes complete files and reconstructs every byte. It uses bounded content boundary searches, stores repeated chunks once, and encodes each new chunk once. On all 25 Canterbury and Calgary files, complete encode-plus-restore CPU time fell 42.7% against a matched whole-file cached store. With two constructed edited versions, CPU time fell 68.3%. The single-version archive is 17.2% larger; these are CPU measurements for this workload, not energy measurements or a universal speedup.

```bash
python scripts/fast_store.py pack versions.fst snapshots/original snapshots/edited
python scripts/fast_store.py unpack versions.fst restored
python scripts/benchmark_fast.py /path/to/corpora
```

## GPU computation reuse

The [CUDA reuse path](docs/GPU_COMPUTE.md) computes distinct blocks once and gathers every full output. Supplied [local chunk-cache measurements](docs/GPU_CHUNK_LOOKUP.md) report 7.53% and 13.53% lower complete elapsed time for two constructed updates, while cold construction is 24.59% slower. A complete lifecycle advantage is not established. The local optimization code and raw timings are not yet integrated; public readiness remains on hold.

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
