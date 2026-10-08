# Dynamic windows and frequent-value removal

This experiment implements Dustin's combined proposal: vary windows according to local value spread, and remove frequent values when describing their placement costs less than retaining them. There is no frequency threshold beyond requiring repetition.

Each window compares raw bytes, integer-mean/signed-offset encoding, and extracting one repeated value into a bitmap plus encoding the remaining values. Candidate extraction costs are computed from one value-count table; only the winning mask is constructed. Remaining values can themselves use mean offsets. Current extraction removes one value per window, not recursive multiple-value dictionaries.

Starting from 1,024-byte contiguous windows, recursive binary splits down to 16 bytes compete against unsplit encoding. Decisions minimize actual bytes including every window header, bitmap, mean and offset width. This is a restricted partition search, not globally optimal boundaries or a full variable-shape 2D search. Local spread affects offset width directly; Shannon entropy is not used as a synonym for numerical distance. The container stores the whole object raw if the window encoding is larger. Raw mode adds five bytes.

## Measurements

All sizes below include the serialized container. Four variants isolate splitting and extraction. Input samples are generated deterministically; these are controlled synthetic cases, not production workload results.

| Sample | Raw bytes | Fixed means/raw | Dynamic means/raw | Fixed + extraction | Dynamic + extraction | zlib |
|---|---:|---:|---:|---:|---:|---:|
| Alternating 0/255 | 65,536 | 65,541 | 65,541 | 8,781 | 8,781 | 86 |
| Ranges changing every 256 bytes | 65,536 | 65,541 | 42,095 | 65,541 | 41,086 | 41,096 |
| Frequent scattered outlier | 65,536 | 65,541 | 65,541 | 20,328 | 20,316 | 21,582 |
| Constant bytes | 65,536 | 461 | 461 | 461 | 461 | 85 |
| Random bytes | 65,536 | 65,541 | 65,541 | 65,541 | 65,541 | 65,562 |
| Synthetic text | 112,445 | 112,450 | 112,450 | 108,143 | 108,143 | 9,357 |

Every variant reconstructs every input exactly. Splitting helps when ranges change inside a window. Extraction helps when a frequent outlier otherwise forces wide offsets. Their combination beats zlib on the constructed outlier sample, while repetition-rich text remains much better suited to zlib. These selected examples do not establish general compression superiority.

The Python search still costs CPU and recursively scans regions. It does not establish a speed or energy advantage; encode timings are recorded in [adaptive.json](../results/lossless/adaptive.json). The next engineering question is choosing candidates cheaply for representative workloads, and considering run/gap encodings alongside bitmaps.

```bash
python scripts/benchmark_adaptive.py
python -m unittest discover -s tests -v
```

25 tests pass. New checks cover exact arbitrary-byte reconstruction, dynamic splitting, extraction, unchanged-object fallback, malformed streams and size bounds.
