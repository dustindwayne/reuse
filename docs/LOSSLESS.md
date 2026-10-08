# Lossless shared structure foundation

The current direction is exact reconstruction from shared definitions. Every distinct byte has one terminal definition. Repeated adjacent pairs become shared nodes pointing at earlier definitions. An ordered root sequence preserves all occurrences. Expanding these references reconstructs the original bytes exactly: case, Unicode encoding, spacing, punctuation and document boundaries are retained. Pair nodes represent repeated byte sequences, not proven linguistic or semantic units.

Definitions and occurrences are different. A symbol is defined once; references still preserve where it occurs. Repeated sequences can collapse those references recursively. Node-length analysis evaluates each definition once and reuses child results. Encoding currently scans the input repeatedly to discover pairs; it is offline preprocessing, not a claim of constant-time ingestion or neural inference savings. Decoding necessarily writes every original byte. Exact reconstruction does not establish semantic understanding.

## Measured storage

Full serialized sizes include header, terminal definitions, pair definitions, and root references. Measurements are in [benchmark.json](../results/lossless/benchmark.json).

| Input | Raw bytes | Grammar bytes | zlib bytes | Terminal definitions | Pair definitions | Root references |
|---|---:|---:|---:|---:|---:|---:|
| 950,000 copies of `s` | 950,000 | 58 | 944 | 1 | 18 | 14 |
| Mixed case, whitespace and Unicode | 19 | 42 | 27 | 16 | 0 | 19 |
| Synthetic training corpus | 112,445 | 31,691 | 9,357 | 25 | 103 | 31,451 |

All three inputs reconstruct byte for byte. Short or irregular inputs can grow. This greedy pair grammar does not find the globally smallest representation, and zlib compresses the synthetic corpus substantially better. The repeated-single-symbol example demonstrates sharing, not superiority on general text.

## Run

```bash
python scripts/benchmark_lossless.py
python scripts/lossless.py pack data/train.txt train.lsg
python scripts/lossless.py unpack train.lsg restored.txt
cmp data/train.txt restored.txt
```

The reader rejects forward/cyclic references, invalid roots and malformed/truncated encodings. Tests include arbitrary bytes, Unicode, large repetition, deterministic output and exact reconstructed lengths. Run `python -m unittest discover -s tests -v` (19 tests).

## Relation to the earlier experiment

The earlier 24-feature representation was lossy and its neural model still predicted complete tokens. Its recorded results are historical evidence for that implementation only. This new module is a lossless foundation; it is not yet connected to a predictor or residual neural model. No new language-model quality or speedup is claimed. The next experimental stage must predict over shared structures without expanding/reanalyzing their definitions, then measure residual-only neural work against a matched baseline. Learned rules must be fit on training data only; using this file compressor on evaluation input is not a valid predictive evaluation.
