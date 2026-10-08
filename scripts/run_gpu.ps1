param(
    [string]$Python = 'python',
    [int]$Repeats = 21,
    [int]$Width = 2048,
    [int]$BatchBlocks = 1024
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    # Use the user's working CUDA environment; do not replace Torch or drivers.
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is unavailable'; print(torch.__version__, torch.cuda.get_device_name())"
    if ($LASTEXITCODE -ne 0) { throw 'Run this in your existing CUDA-enabled Python environment.' }
    & $Python scripts/download_corpora.py data/corpora
    if ($LASTEXITCODE -ne 0) { throw 'Corpus download or hash verification failed.' }
    & $Python scripts/benchmark_gpu_lifecycle.py data/corpora --repeats $Repeats --width $Width --batch-blocks $BatchBlocks
    if ($LASTEXITCODE -ne 0) { throw 'GPU lifecycle validation failed.' }
    Write-Host 'Lifecycle measurement completed. Review paired savings and win counts in results/lossless/gpu_lifecycle.json. Success means exact measurement completed, not a guaranteed speedup.'
} finally {
    Pop-Location
}
