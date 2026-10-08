# Integrated lossless data container

One file container combines cross-file shared chunks and content-dependent encoding. File extensions do not select codecs. Each distinct chunk is encoded once; files store ordered references. Hashes locate candidates, then byte equality verifies sharing. Reading decodes each unique chunk once and reuses it for reconstruction.

For each unique chunk, the encoder compares raw bytes, zlib, dynamic average/offset windows with repeated-value extraction, the shared byte grammar, and fixed grid averages, plus each structural transform followed by zlib. All mode/header sizes are equal per chunk, so minimum payload size selects the codec. zlib is an explicitly included baseline and fallback, not a new algorithm attributed to this project. The manifest preserves names, sizes, checksums and reconstruction order. This prototype preserves file contents and names, not permissions, timestamps, filesystem links or application semantics.

## Results

| Scenario | Original bytes | Container bytes | ZIP bytes |
|---|---:|---:|---:|
| Mixed text, patterns, values and random bytes | 161,597 | 41,443 | 31,575 |
| Shared chunks across three files | 135,168 | 17,050 | 50,212 |
| Unique random bytes | 65,536 | 66,363 | 65,674 |

Sizes include every dictionary entry, checksum, file reference and header. Exact reconstruction passes for all cases. The sharing scenario encodes five unique chunks for 33 references. Mixed-data selection uses direct zlib, grammar followed by zlib, and raw storage. The shared-file sample also selects dynamic encoding followed by zlib. Full timings and counts: [container.json](../results/lossless/container.json).

Sharing gives a demonstrated benefit in the constructed cross-file duplication case. The container loses to ZIP on mixed data and incurs metadata on random data. The Python encoder evaluates all candidates and is expensive; these results do not show reduced power consumption or general superiority. Current outer chunks have fixed 4,096-byte boundaries, so shifted repetitions can be missed. Dynamic splitting happens inside candidate payloads. The archive is an integrated storage prototype; it does not yet implement predictive learning, residual-only neural computation, direct application execution, or persistent cross-archive sharing.

## Run

```bash
python scripts/benchmark_container.py
python scripts/container.py pack example.udf data/train.txt data/validation.txt
python scripts/container.py unpack example.udf restored
cmp data/train.txt restored/train.txt
cmp data/validation.txt restored/validation.txt
```

Packing refuses an existing output file. Extraction requires a new output directory and rejects unsafe paths. Contents and references are checked before writing files. This is a prototype format for locally generated archives; resource limits for hostile compressed archives are not implemented.

28 unit tests pass, including shared-file reconstruction, empty objects, Unicode names and corruption checks.

Codec identifiers: 0 raw, 1 zlib, 2 dynamic, 3 grammar, 4 grid, 5 dynamic then zlib, 6 grammar then zlib, 7 grid then zlib. See [composition measurements](COMPOSED.md).
