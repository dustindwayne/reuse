# Structural Residual LM

An experimental investigation into whether predictable linguistic structure can be handled deterministically so neural computation can focus on residual uncertainty.

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

## Why this exists

Earlier work explored selective access to large models, submarine-window activity and trie-based structure learning. The new experiment asks whether deterministic prediction and observed residual uncertainty can guide neural computation. Original chat records were not recovered; [original notes](docs/ORIGINAL_NOTES.md) preserve reported wording with explicit secondary-source labels. The [timeline](docs/TIMELINE.md) separates supported artifact dates from unresolved search leads.

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
