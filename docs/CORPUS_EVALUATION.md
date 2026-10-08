# Real-corpus evaluation

Tested the complete 11-file Canterbury corpus and standard 14-file Calgary corpus: 25 files, 5,952,406 bytes. These contain real text, source, markup, spreadsheet, executable, image and geophysical data. No samples were truncated. Inputs match the pinned mirror Git blobs and its published SHA1 checksums; SHA-256 values are recorded in [corpus_sources.json](../results/lossless/corpus_sources.json). Raw corpus files are not redistributed in this repository.

Sources: [official descriptions](https://corpus.canterbury.ac.nz/descriptions/) and [pinned mirror](https://github.com/pfalcon/canterbury-corpus/tree/master). The mirror tree is `c380ef9dc1dc62691757fdfeebdc2f85710e1e46`; the manifest pins each blob rather than depending on the mutable branch.

## Whole-file transforms followed by zlib

Each complete file compares raw, zlib, dynamic windows, shared grammar (32 merges), grid, and each transform followed by zlib. Size selection counts internal transform metadata. These payload totals omit common archive framing, so they isolate composition rather than claiming complete-archive superiority. One encode/decode timing per whole file; exact byte equality is asserted.

| Corpus | Direct zlib payload bytes | Selected payload bytes | Reduction | Improved files |
|---|---:|---:|---:|---:|
| canterbury | 730,954 | 711,035 | 2.73% | 6/11 |
| calgary | 1,025,183 | 997,113 | 2.74% | 11/14 |

## Complete archives and matched controls

Framework archives include all metadata, checksums and references. The matched control has the same container, sharing and chunk size but only raw/zlib candidates. ZIP uses per-file DEFLATE. This isolates transforms from sharing and chunking. Archive framework timings have one repeat at each chunk size; ZIP and the 4 KB matched control have three. The 32 KB matched control has one.

| Corpus | Framework 4 KB | Matched 4 KB | Framework 32 KB | Matched 32 KB | ZIP |
|---|---:|---:|---:|---:|---:|
| canterbury | 904,037 | 907,005 | 755,888 | 771,860 | 731,938 |
| calgary | 1,329,586 | 1,337,698 | 1,090,842 | 1,109,463 | 1,026,311 |

## Every file, including standard baselines

Per-file framework and matched-control sizes include a one-file archive. zlib/gzip/bzip2/xz sizes include their own stream headers but no filename/manifest. Per-file times are medians of three serial runs; all compressed bytes are deterministic and all methods pass exact reconstruction. Defaults: zlib level 6, gzip level 9, bzip2 level 9, xz preset 6, ZIP DEFLATE default.

| File | Raw | Framework 4 KB | Matched 4 KB | zlib | gzip | bzip2 | xz | Whole-file selected mode | Whole-file selected bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| calgary/bib | 111,261 | 53,509 | 53,509 | 35,228 | 35,083 | 27,467 | 30,588 | grammar → zlib | 33,401 |
| calgary/book1 | 768,771 | 401,232 | 401,232 | 313,582 | 312,498 | 232,598 | 261,116 | grammar → zlib | 306,209 |
| calgary/book2 | 610,856 | 282,932 | 282,932 | 206,664 | 206,147 | 157,443 | 169,824 | grammar → zlib | 200,211 |
| calgary/geo | 102,400 | 68,521 | 73,852 | 68,433 | 68,373 | 56,921 | 53,364 | dynamic → zlib | 63,791 |
| calgary/news | 377,109 | 187,611 | 187,611 | 144,800 | 144,492 | 118,600 | 118,900 | grammar → zlib | 142,057 |
| calgary/obj1 | 21,504 | 11,248 | 11,248 | 10,317 | 10,329 | 10,787 | 9,428 | zlib | 10,317 |
| calgary/obj2 | 246,814 | 104,459 | 104,459 | 81,505 | 81,027 | 76,441 | 61,504 | zlib | 81,505 |
| calgary/paper1 | 53,161 | 24,420 | 24,420 | 18,558 | 18,536 | 16,558 | 17,280 | grammar → zlib | 18,282 |
| calgary/paper2 | 82,199 | 39,251 | 39,251 | 29,760 | 29,677 | 25,041 | 27,228 | grammar → zlib | 29,148 |
| calgary/pic | 513,216 | 63,945 | 66,718 | 56,465 | 52,233 | 49,759 | 41,992 | dynamic → zlib | 53,384 |
| calgary/progc | 39,611 | 17,312 | 17,312 | 13,343 | 13,342 | 12,544 | 12,560 | zlib | 13,343 |
| calgary/progl | 71,646 | 22,872 | 22,872 | 16,255 | 16,158 | 15,579 | 14,984 | grammar → zlib | 16,178 |
| calgary/progp | 49,379 | 16,507 | 16,507 | 11,228 | 11,180 | 10,710 | 10,352 | grammar → zlib | 10,850 |
| calgary/trans | 93,695 | 35,267 | 35,275 | 19,045 | 18,928 | 17,899 | 16,720 | grammar → zlib | 18,437 |
| canterbury/alice29.txt | 152,089 | 72,234 | 72,234 | 54,404 | 54,182 | 43,202 | 48,492 | grammar → zlib | 52,789 |
| canterbury/asyoulik.txt | 125,179 | 62,419 | 62,419 | 48,897 | 48,790 | 39,569 | 44,536 | grammar → zlib | 47,454 |
| canterbury/cp.html | 24,603 | 10,483 | 10,483 | 7,961 | 7,952 | 7,624 | 7,644 | grammar → zlib | 7,819 |
| canterbury/fields.c | 11,150 | 3,904 | 3,904 | 3,122 | 3,127 | 3,039 | 3,028 | zlib | 3,122 |
| canterbury/grammar.lsp | 3,721 | 1,398 | 1,398 | 1,222 | 1,234 | 1,283 | 1,292 | zlib | 1,222 |
| canterbury/kennedy.xls | 1,029,744 | 223,968 | 223,968 | 203,992 | 207,041 | 130,280 | 49,116 | zlib | 203,992 |
| canterbury/lcet10.txt | 426,754 | 199,762 | 199,762 | 144,904 | 144,451 | 107,706 | 119,500 | grammar → zlib | 138,138 |
| canterbury/plrabn12.txt | 481,861 | 247,645 | 247,651 | 195,261 | 194,344 | 145,577 | 165,412 | grammar → zlib | 188,389 |
| canterbury/ptt5 | 513,216 | 63,946 | 66,719 | 56,465 | 52,233 | 49,759 | 41,992 | dynamic → zlib | 53,384 |
| canterbury/sum | 38,240 | 15,925 | 16,114 | 12,990 | 12,850 | 12,909 | 9,452 | zlib | 12,990 |
| canterbury/xargs.1 | 4,227 | 1,986 | 1,986 | 1,736 | 1,748 | 1,762 | 1,812 | zlib | 1,736 |

## Compute cost

| Corpus | Sum of per-file framework encode medians | Sum of matched encode medians | Framework decode medians | Matched decode medians |
|---|---:|---:|---:|---:|
| canterbury | 46.0255 s | 0.1432 s | 0.0792 s | 0.0345 s |
| calgary | 76.0041 s | 0.1382 s | 0.1074 s | 0.0355 s |

## What the evidence establishes

Transform composition can improve direct zlib on real complete files. The current complete archive still loses to ZIP on both corpora. The 4 KB framework is especially constrained by lost compression context and metadata. Larger chunks improve archive size and allow more structural candidates to win. These measurements do not show a general compression breakthrough or lower power consumption: the Python encoder searches multiple candidates and is much slower than direct zlib. Whole-file compression and shared storage have different access/sharing tradeoffs.

Corpus parameters were chosen before these results, not tuned per file. Standard corpora are small, old benchmarks, not a substitute for modern multimedia, encrypted data, application access or multi-version storage tests. Corpus sizes are storage bytes, not linguistic token counts. The framework preserves exact bytes, not a learned semantic model.

Process peak RSS in the main runner is a cumulative high-water mark, not per-codec memory. Timings are serial measurements on this host, without confidence intervals; whole-file and archive sensitivity timing has only one repeat. Tests include ordinary natural text and binary data, but no claim of universal improvement follows.

29 unit tests pass, including exact reconstruction, shared references, malformed streams and the matched-control selector.

Full records: [per-file/archive tests](../results/lossless/corpora.json), [whole-file tests](../results/lossless/corpora_whole.json). Environment and repeated timings are in those files.

## Reproduce

```bash
python scripts/download_corpora.py /tmp/corpus-bench
python scripts/benchmark_corpora.py --corpus-root /tmp/corpus-bench --output results/lossless/corpora.json
python scripts/benchmark_corpora_whole.py /tmp/corpus-bench results/lossless/corpora_whole.json
python scripts/report_corpora.py
python -m unittest discover -s tests -v
```

The main runner checkpoints after each file. Delete its output to rerun all timings; otherwise it resumes files whose input hashes and algorithm-source hashes match. Dataset download uses public GitHub blob endpoints and verifies pinned SHA-256 values.
