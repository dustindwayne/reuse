# Complete data with less CPU work

The new path reduces measured CPU time for complete lossless encoding and reconstruction on the pinned Canterbury and Calgary corpora. It replaces the exhaustive candidate search with one encoding per new chunk and reuses already encoded chunks across files and versions. Nothing is discarded or approximated.

## Complete-input measurements

25 complete source files contain 5,952,406 bytes. The incremental experiment adds two constructed versions of every source file: an insertion at one-third, followed by a replacement at two-thirds. It processes and reconstructs all 17,858,293 bytes across all three versions. These modifications are a controlled workload over real corpus files, not a recorded real editing history.

Seven repetitions, median process CPU seconds, same zlib level 6, same interpreter and machine:

| Method | Original encode + restore CPU | Three-version encode + restore CPU | Original stored bytes | Three-version stored bytes |
|---|---:|---:|---:|---:|
| Matched whole-file cached store | 0.3195 | 0.8674 | 1,703,941 | 5,113,367 |
| Fixed 64 KiB cached chunks | 0.2690 | 0.6122 | 1,763,145 | 3,949,220 |
| Content boundary cached chunks, default | **0.1831** | **0.2752** | 1,997,114 | 2,410,403 |
| Content boundaries with sampled folding | 0.1961 | 0.2782 | 1,997,114 | 2,410,403 |
| Direct codec, without store metadata or checksums | 0.3038 | 0.8621 | 1,756,137 | 5,270,085 |

The default uses **42.7% less CPU** on the original complete dataset and **68.3% less CPU** over the three-version lifecycle than the matched cached whole-file store. Against direct codec-only work, reductions are 39.7% and 68.1%. The default three-version measurements range from 0.2642 to 0.2967 seconds; matched whole-file caching ranges from 0.8266 to 0.9448 seconds. Wall times and individual repetitions are in [the result file](../results/lossless/fast_compute.json).

Store measurements include boundary scanning, chunk hashing, whole-file hashing, exact duplicate checks, compression, pointer manifests, serialization, decoding every distinct chunk, reconstruction of every full version, and integrity verification. The input files are loaded before timing. Disk/network transfer, process startup and the final benchmark assertion are excluded. Decode integrity checks are included; the assertion independently verifies all restored contents. Median encode and decode values are reported separately in the JSON; their individual medians need not add to the median total.

The matched whole-file baseline uses the same store format, checksums, caching and restore code. It reuses identical files, including the identical `ptt5` and `pic` inputs. The direct-codec row omits these store costs and encodes each file version independently. It is a simpler favorable CPU comparison, not a complete serialized archive.

## Why work decreases

The boundary finder searches byte patterns in native regex code after an 8 KiB minimum and stops at a 64 KiB maximum. A double newline or adjacent bytes from `00`, `20`, `80`, `ff` can establish a cut. These are inexpensive byte heuristics, not semantic parsing, an entropy estimator or a rolling fingerprint. They can realign after inserted data, allowing later chunks to match previous chunks.

A SHA-256 lookup finds an existing chunk; a full byte comparison confirms equality. Reused chunks receive pointers and skip compression entirely. New chunks receive one level-6 encoding, or raw storage if compression would enlarge the chunk. The decoder reconstructs each distinct chunk once and reuses its bytes when rebuilding files.

Across the three versions, the default encodes 7,026,079 distinct input bytes and reuses 10,832,214 bytes. The whole-file cached baseline encodes 16,318,602 bytes. All input bytes are still inspected and every output byte is reconstructed. Reuse skips expensive encoding work; it does not eliminate reads, hashing or output copies.

Even on a single version, shorter compression histories reduce encoding work on this dataset. That saves CPU at the cost of compression density: the default single-version archive is **17.2% larger** than the whole-file cached archive, though the three-version archive is 52.9% smaller. Storage size is a separate measurement from the compute result.

The optional sampled folding selector tries four encodings of a 2 KiB prefix on sufficiently large new chunks, then encodes the full chunk only once. It did not improve these stored sizes and increased single-version CPU, so it is disabled by default. The expensive grammar/grid/multi-path selection remains available as earlier experiments but is absent from the default fast path.

## Use and reproduce

```bash
python scripts/download_corpora.py --help
python scripts/benchmark_fast.py /path/to/corpora --repeats 7
python scripts/fast_store.py pack versions.fst snapshots/original snapshots/edited
python scripts/fast_store.py unpack versions.fst restored
```

Each source directory becomes a named version; files use relative paths. Versions share chunks in one `FST1` store. The CLI refuses an existing output and verifies the stored data when restoring. Directory names must be distinct. Empty files reconstruct exactly. Symlinks are unsupported; file permissions, timestamps and empty directories are not represented.

For an in-process incremental pipeline:

```python
from structural_residual_lm.fast import Store, loads

store = Store()  # one compression per new chunk; no transform search
store.add("original", {"data.bin": original_bytes})
store.add("edited", {"data.bin": edited_bytes})
encoded = store.dumps()
assert loads(encoded)["edited"]["data.bin"] == edited_bytes
```

The encoder keeps original distinct chunks in memory for exact duplicate confirmation. This prototype is an in-memory batch/incremental store, not yet a disk-backed cache or a bounded-memory streaming system. Timings do not establish electricity savings, faster neural inference or a universal improvement for every data distribution. They establish reduced CPU time for full lossless data processing on the measured complete-file workloads. Different boundary rules and repetition rates can change the outcome.

Validation: the complete test suite passes (41 tests), including shifted-insertion reuse, exact version reconstruction, bounded coverage and corruption rejection. A CLI pack/unpack check independently restores two full directory versions and empty files exactly.
