# Structural transforms followed by zlib

The framework now tests composition, not only selection between methods: dynamic windows with frequent-value extraction, shared grammar and fixed grid transforms can each feed zlib. Decode reverses zlib first, then the structural transform. The container chooses the smallest actual payload, including any internal metadata, among raw, direct zlib, each transform alone and each transform followed by zlib.

## Whole-object measurements

| Sample | Direct zlib | Dynamic then zlib | Grammar then zlib | Grid then zlib |
|---|---:|---:|---:|---:|
| Changing ranges | 41,096 | 38,330 | 41,317 | 42,182 |
| Frequent outlier | 21,582 | 18,955 | 17,298 | 22,724 |
| Alternating extremes | 86 | 79 | 47 | 274 |
| Random bytes | 65,562 | 65,567 | 80,252 | 66,579 |
| Synthetic text | 9,357 | 18,154 | 8,178 | 40,653 |

All sizes are real serialized bytes, including transform metadata and zlib headers, excluding the common outer container. Exact roundtrips passed. Full records: [composed.json](../results/lossless/composed.json). These are controlled samples; results vary by data and transform.

The container operates on 4,096-byte shared chunks with a 32-merge grammar candidate. This whole-object benchmark uses up to 128 grammar merges, so it does not imply identical archive compression. Integrated archive measurements are recorded separately in [CONTAINER.md](CONTAINER.md). Chunk boundaries trade compression context against sharing and access; optimizing that trade remains unfinished.

```bash
python scripts/benchmark_composed.py
python scripts/benchmark_container.py
python -m unittest discover -s tests -v
```

28 tests pass, including exact decoding of all three transform-plus-zlib modes. No speed or energy improvement is claimed from these storage measurements.
