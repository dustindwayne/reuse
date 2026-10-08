# Changelog

## 2026-10-08

- Roll back appended cache definitions on failed processing and publish outputs only after reconstruction succeeds; add failure/retry regression coverage.
- Reject adaptive windows larger than 65,534 bytes and require setuptools 77 for SPDX license metadata.
- Preserve original measured sources, scrub personal machine paths from docs/session metadata, and correct current CI/test/baseline descriptions.

- Remove unsupported conversation quotations, origin timelines, source-recovery pointers, and reported historical results without committed evidence.

- Isolate the earlier language-model source, scripts, configs, smoke data, tests, and results under `archive/language-model/`; retain the evidence trail.
- Split CI into active and historical test jobs, update package metadata to reuse, and resolve moved source paths when checking the original GPU freeze manifest.
- Keep current root architecture/results focused on complete-data storage and computation reuse.

- Add exact chunk-level persistent GPU reuse, verified chunk lookup, contiguous group packing and identity-gather bypass.
- Preserve complete local raw measurements, earlier negative results and three independent frozen lifecycle sessions.
- Make complete cold-plus-update lifecycle elapsed time the primary GPU metric: two-edit gains are unstable; four/eight constructed edits improve across three sessions.
- Update reproduction commands and add clean-checkout result/freeze/CUDA exactness verification without benchmark expansion.

## 2026-10-07

- Implement deterministic compact encoder and context trie.
- Establish neural baseline before structural integration.
- Add causal residual representation, bounded surprise and actual conditional block execution.
- Correct throughput measurement to include structural overhead.
- Run three configurations and 14 conditions across three seeds.
- Preserve negative performance results.
