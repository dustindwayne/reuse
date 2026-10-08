# Pattern matching before chunk boundaries

The pattern-first path searches exact repeated sequences over the concatenated input files before encoding new literal data. A match becomes a backward range reference (source offset and length), including matches across files and positions shifted by insertions. New bytes enter one literal pool. The remaining framework chooses a codec for that pool; the manifest, references and encoded pool can then pass through zlib together. Reconstructed file names, lengths and SHA-256 values preserve file identity.

The current matcher uses 16-byte sparse anchors, queries every current byte position, retains at most eight prior candidates per key, requires a 32-byte minimum match and limits matches to 65,536 bytes. It verifies bytes exactly, greedily takes the longest candidate, and permits only fully earlier ranges. It is a bounded search, not an exhaustive optimal dictionary. Match length currently approximates reference savings before compression; it does not optimize global post-zlib size. Parameters were fixed before evaluating the corpora.

## Measurements

| Input | Input bytes | Pattern-first archive | Same pipeline without matching |
|---|---:|---:|---:|
| Original, shifted and edited random-byte file | 98,323 | 33,146 | 98,666 |
| Full Canterbury corpus | 2,810,784 | 639,958 | 632,367 |
| Full Calgary corpus | 3,141,622 | 1,034,762 | 1,028,367 |

The shifted/edited test is explicitly synthetic. Its fixed-chunk archive is 91,541 bytes. Pattern-first matching reuses 65,536 bytes through three range references. It demonstrates position-independent reuse rather than treating these invented file sizes as user claims.

On real Canterbury inputs, 440,281 bytes become 5,711 references; on Calgary, 615,924 bytes become 8,273 references. Those references plus changed literal ordering make the final archives 1.20% and 0.62% larger than the same pipeline without matching. These controls include exactly the same outer compression and metadata. They isolate matching rather than comparing unrelated formats. The existing zlib step already captures many local repetitions, and splitting them into a separate reference stream can weaken its context.

Exact reconstruction passes for all three sets. Encoding takes about 16 seconds for Canterbury and 19 seconds for Calgary on this host; timings are one repeat, not statistically precise. Complete counts, mode choices, instruction sizes and timings are in [patterns.json](../results/lossless/patterns.json). The corpus inputs are the same pinned and checksum-verified originals documented in [CORPUS_EVALUATION.md](CORPUS_EVALUATION.md).

32 unit tests pass, including shifted insertion/deletion-boundary reuse, arbitrary bytes, empty files, Unicode, seeds with different continuations, and corrupt streams. Decoder integrity checks preserve exact bytes; the prototype is not hardened against hostile resource-exhaustion inputs.

## Run

```bash
python scripts/container.py pack --patterns example.ptf data/train.txt data/validation.txt
python scripts/container.py unpack example.ptf restored
python scripts/benchmark_patterns.py /tmp/corpus-bench
```

Extraction preserves file contents and names, not permissions or timestamps. The CLI refuses existing destinations and validates paths before extraction.

The next change should be selective pattern admission by final encoding cost, rather than applying every sufficiently long match. Keeping a pattern once is useful only if references and altered compression context produce a smaller complete representation.
