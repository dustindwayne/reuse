param(
    [string]$Python = 'python',
    [int]$Repeats = 7,
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
    & $Python scripts/benchmark_gpu.py data/corpora --repeats $Repeats --width $Width --batch-blocks $BatchBlocks
    if ($LASTEXITCODE -ne 0) { throw 'GPU improvement gate did not pass. See results/lossless/gpu_compute.json.' }
    Write-Host 'GPU measurement completed. Review results/lossless/gpu_compute.json before considering publication.'
} finally {
    Pop-Location
}
