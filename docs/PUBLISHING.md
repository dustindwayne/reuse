# Private GitHub repository

The complete project is available at [dustindwayne/reuse](https://github.com/dustindwayne/reuse). The repository is private.

All 14 development snapshots were imported in order after the repository bootstrap commit, including code, documentation, benchmark summaries and synthetic per-example CSV logs. Every imported snapshot was verified against its original Git tree SHA. The final imported tree is `9e9d48a19ebd9dd8bcb7b00a4094d9e39b569ccd`.

GitHub recreated commit identifiers and timestamps. [IMPORT_MAP.json](IMPORT_MAP.json) maps original local commit identifiers, including those recorded in benchmark metadata, to their GitHub equivalents. The previously delivered Git bundle retains the original history and timestamps.

The repository includes a GitHub Actions workflow that installs dependencies in a fresh hosted runner, executes unit tests, regenerates the corpus and runs all three benchmark configurations. See [VALIDATION.md](VALIDATION.md) for observed checks.

Current status: the tested GPU chunk cache, raw session records and source-freeze manifest are integrated with the [frozen GPU lifecycle reports](GPU_LIFECYCLE.md). Four/eight constructed edits show repeated scoped elapsed-time gains; two edits remain unreliable. Keep the repository private pending review of the scoped claim. Historical quotations and dates must remain explicitly sourced; they are not evidence for current performance. Integration does not authorize public release.
