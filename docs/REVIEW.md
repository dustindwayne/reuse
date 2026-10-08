# Current review status

## Addressed

- Failed cache processing rolls back new block/chunk definitions and lookup entries. Cached outputs are published only after full output return and reconstruction succeed. Regression coverage checks raised kernels, bad output counts, retries, and later reuse for both cache types without requiring Torch, plus a real Torch failure/retry test that uses CUDA when available.
- Adaptive windows are limited to 65,534 bytes so the raw encoding plus its mode byte fits the 16-bit payload-length field. The format is unchanged.
- Packaging requires setuptools 77 and offers an optional `gpu` dependency. The tested CUDA installation remains environment-specific; the extra is not a pinned CUDA reproduction environment.
- Current test counts, CI behavior, matched-layout baseline, same-machine session limits, memory retention, and post-measurement source changes are explicit. Personal executable/repository paths were removed from docs and recorded command metadata; measurement values, source hashes, and session logs were preserved.

## Remaining limits

Decoders have no configured total output/allocation limit. Length and integrity checks do not prevent compressed or grammar payloads from consuming excessive memory before rejection. Current archive decoding is intended for trusted inputs. Supporting externally supplied archives requires bounded decompression, grammar expansion checks before materialization, and manifest/count budgets across nested formats; this review patch does not claim that hardening is complete.

Cache identity remains caller-supplied. Changes to operation parameters, closures, or weights require a new identity; Python code hashing alone would not establish semantic identity. Capacity eviction and reduced input/output retention are not implemented.

The retained three sessions are sequential processes on one machine. No confidence interval accounting for serial correlation or randomized-order experiment is available. Expensive-kernel crossover, low-reuse workloads, and bounded-capacity behavior remain untested. No new timing experiment was added for this review.

Original raw source hashes remain byte-sensitive. `.gitattributes` preserves their recorded line endings. Changed measured files are preserved under [archive/measured-gpu](../archive/measured-gpu/README.md), and the original manifest resolves via `archive/path-map.json`. This verifies historical bytes; current robustness fixes need new CUDA exactness checks and measurements before inheriting performance claims.

The existing import namespace, archived CSV volume, and compact source style remain unchanged. These are maintenance choices rather than new performance evidence.
