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

Remaining checks before public publication: primary chat quotation/date verification, full isolated dependency installation, GitHub remote creation and remote CI execution. Local source and full development history are delivered in the archive; no private GitHub repository has yet been created.
