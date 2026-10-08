# Earlier language-model experiment

Historical experiment: structural prediction and conditional neural blocks over a synthetic grammar corpus. It skipped optional block work but did not reduce complete elapsed time. This is not the active reuse implementation.

The source, tests, configs, data, and raw results are preserved here. Run its commands from this directory so relative configuration and output paths remain valid:

```bash
cd archive/language-model
python -m pip install -r requirements-reproduce.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/prepare_data.py
python scripts/benchmark.py --config configs/baseline.yaml
```

On PowerShell, set `$env:PYTHONPATH = 'src'` before running the test command. Use a separate process from active-package tests; the historical and active packages retain the same import namespace.

[Architecture](ARCHITECTURE.md), [results](RESULTS.md), and [historical overview](../../docs/archive/EARLIER_LANGUAGE_MODEL.md) explain the original scope. Recorded sources and result bytes are preserved; links and archive instructions have been updated for the new location.
