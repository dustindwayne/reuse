# Validation

Completed 2026-10-07:

- 16 unit tests pass, including numerical gradient checks, data split disjointness, train-only vocabulary, unknown tokens, trie fallback, residual reconstruction and causal randomized routing.
- 42 final condition/seed combinations: 14 conditions × seeds 7, 11, 23. Six corrected control reruns replace earlier invalid values while Git retains the earlier records.
- Clone the local repository into a clean checkout, build and install its wheel, regenerate the corpus, rerun all three models and print checkpoint next-token predictions.
- Clean-checkout validation NLL reproduces exactly to within 1e-12: baseline 0.781269855256834; structural 0.7728431835192152; residual 0.7729357517793882. Dataset hashes also match.
- All result JSON parses; relative Markdown links resolve; Python sources compile.
- Credential and machine-specific absolute-path scan finds no matches in tracked deliverables.

The checkout test used a new virtual environment with inherited, preinstalled NumPy/PyYAML and an offline wheel build. This validates packaging and code reproducibility, but does not demonstrate fresh network dependency resolution in a fully isolated environment. Exact recorded dependency versions are in `requirements-reproduce.txt`.

Hardware is captured in `results/hardware.json`. Timing and process RSS vary between runs; exact NLL reproduction does not make throughput deterministic. Git dirty flags in some experiment records reflect rewritten tracked results; the recorded implementation commits identify the code used. Later corrections are documented and affected conditions rerun.

GitHub upload completed 2026-10-08: the private repository contains all 14 development snapshots and synthetic per-example CSV logs. Every snapshot tree matches the original local tree; commit identifiers are mapped in [IMPORT_MAP.json](IMPORT_MAP.json).

[CI run 37721547238](https://github.com/dustindwayne/structural-residual-lm/actions/runs/37721547238) passed on a fresh GitHub-hosted Ubuntu runner using Python 3.12. It installed dependencies, passed the 16 unit tests, regenerated data and ran baseline, structural and residual benchmarks. Hosted dependency resolution therefore succeeds; this workflow does not assert bitwise loss reproduction across dependency versions or hardware. The imported final snapshot also triggers the same workflow; current status is available in [Actions](https://github.com/dustindwayne/structural-residual-lm/actions).

Remaining check before public publication: primary chat quotation/date verification. The archive and Git bundle retain the original local development history.
