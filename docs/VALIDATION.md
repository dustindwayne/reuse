# Validation

## Local CUDA review of the current fixes

The local runner checked commit `59cb8feb1c5222b02028c15c29ef7e7da5ac48ef`: all 36 active and 16 archived tests passed with no skips. Both fixture and full-corpus CUDA exactness checks passed nine versions on the RTX 5070 Ti. Historical measured-source hashes and all 252 recorded statistics verified. No new timing records were produced. Three checkout paths have a documented CRLF clean-filter issue despite matching committed physical bytes; this review does not claim a clean Git status. See [the local review](LOCAL_REVIEW_59cb8feb.md).

## Earlier checks in this workspace

Before the archive split, the integrated suite had 49 passing unit tests. The split passes locally: 36 active tests (three optional-Torch tests skipped here) and 16 historical language-model tests in a separate process. All original freeze hashes and recorded paired statistics still verify. Cache rollback and adaptive-window regression tests were added after the split. No GPU timing was rerun for these changes. All 252 paired complete GPU lifecycles and 1,197 bitwise-output / byte-exact reconstruction comparisons across three frozen sessions are preserved with the raw JSON, logs and source-freeze manifest. The verification command checks those hashes and recalculates recorded medians, paired savings and win counts. CUDA integration verification covers all nine existing constructed versions of the complete pinned corpora, without adding timing experiments. See [current GPU findings](GPU_LIFECYCLE.md) and [verification commands](GPU_COMPUTE.md). The earlier prototype's 45-test stage remains archived.

The CPU shared-store measurements and raw runs are checked in under [fast compute](FAST_COMPUTE.md). Historical language-model validation below is retained as an earlier stage, not the current test count or current performance claim.

## Historical language-model validation

Completed 2026-10-07:

- 16 unit tests pass, including numerical gradient checks, data split disjointness, train-only vocabulary, unknown tokens, trie fallback, residual reconstruction and causal randomized routing.
- 42 final condition/seed combinations: 14 conditions × seeds 7, 11, 23. Six corrected control reruns replace earlier invalid values while Git retains the earlier records.
- Clone the local repository into a clean checkout, build and install its wheel, regenerate the corpus, rerun all three models and print checkpoint next-token predictions.
- Clean-checkout validation NLL reproduces exactly to within 1e-12: baseline 0.781269855256834; structural 0.7728431835192152; residual 0.7729357517793882. Dataset hashes also match.
- All result JSON parses; relative Markdown links resolve; Python sources compile.
- The earlier absolute-path scan is historical; later GPU reports and session metadata contained machine paths, now scrubbed.

The checkout test used a new virtual environment with inherited, preinstalled NumPy/PyYAML and an offline wheel build. This validates packaging and code reproducibility, but does not demonstrate fresh network dependency resolution in a fully isolated environment. Exact recorded dependency versions are in `../archive/language-model/requirements-reproduce.txt`.

Hardware is captured in `../archive/language-model/results/hardware.json`. Timing and process RSS vary between runs; exact NLL reproduction does not make throughput deterministic. Git dirty flags in some experiment records reflect rewritten tracked results; the recorded implementation commits identify the code used. Later corrections are documented and affected conditions rerun.

GitHub upload completed 2026-10-08: the private repository contains all 14 development snapshots and synthetic per-example CSV logs. Every snapshot tree matches the original local tree; commit identifiers are mapped in [IMPORT_MAP.json](IMPORT_MAP.json).

[CI run 37721547238](https://github.com/dustindwayne/reuse/actions/runs/37721547238) passed on a fresh GitHub-hosted Ubuntu runner using Python 3.12. It installed dependencies, passed the 16 unit tests, regenerated data and ran baseline, structural and residual benchmarks. Hosted dependency resolution therefore succeeds; this workflow does not assert bitwise loss reproduction across dependency versions or hardware. The workflow has since been split into active CPU/evidence checks and archived language-model tests; current status is available in [Actions](https://github.com/dustindwayne/reuse/actions).

Optimized source, raw GPU records and the freeze manifest are integrated; release review still requires accurately scoping the workload. The archive and Git bundle retain original development history. Public release is not authorized by source integration.
