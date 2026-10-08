# Local checks for the review fixes

Use the existing RTX 5070 Ti / `torch5070` environment. Test a clean checkout of the review commit; preserve any local edits. This run checks correctness only and produces no new timing records.

From the existing repository in PowerShell:

```powershell
git fetch origin
git worktree add --detach ..\reuse-review-check origin/main
Set-Location ..\reuse-review-check
conda activate torch5070
$env:PYTHONPATH = 'src'
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m unittest discover -s tests -v
python scripts/verify_gpu_integration.py
```

Use an unused worktree path if this one already exists. Confirm `git rev-parse HEAD` is the review-fix commit supplied with the handoff before reporting results.

The active suite now has **36 tests**. All should execute with Torch installed. `test_real_torch_failure_and_retry` uses CUDA when available and checks both block and chunk caches after a raised kernel error and an invalid output count. It then processes different data and retries the failed input, comparing every output against the dense NumPy reference and reconstructing bytes exactly. The adaptive boundary regression rejects 65,535-byte windows and verifies 65,534-byte windows round-trip.

For the existing full-corpus exactness checks, reuse the pinned corpus directory or download it into this checkout:

```powershell
python scripts/download_corpora.py data/corpora
python scripts/verify_gpu_integration.py --corpora data/corpora
```

This verifies full outputs and reconstruction for the existing constructed versions. It also checks preserved original source hashes and recorded statistics; those historical checks do not measure performance of the current fixes.

Historical tests run in a separate process:

```powershell
Push-Location archive/language-model
python -m unittest discover -s tests -v
Pop-Location
```

Expected: 16 historical tests. Report commit SHA, Python/Torch/CUDA versions, GPU name, active and historical test outcomes/skips, and full-corpus verifier output. Include any traceback in full. Do not push, alter visibility, overwrite archived measurements, or run timing/tuning as part of these checks.
