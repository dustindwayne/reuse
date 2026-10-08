# GPU computation reuse: implementation ready, measurements pending

Status: **no GPU improvement has been measured**. The current Work session exposes a cloud browser and a CPU workspace, not the user's desktop GPU. PyTorch is absent in the workspace. The benchmark records `status: blocked` instead of silently substituting CPU results. Keep the repository private pending actual GPU measurements and review.

## Implemented path

`BlockPlan` covers every byte of every input file. It finds the same bounded content boundaries as the fast store, splits each chunk into fixed-width local blocks, verifies duplicate bytes, and records references. It retains exact original lengths, file hashes and version manifests. Tail blocks are zero-padded only for computation; byte reconstruction uses their original contents.

`execute_torch` sends distinct blocks to CUDA, evaluates a deterministic block-local kernel once for each distinct input, and gathers the complete output matrix for every original occurrence. It batches at a fixed cap to limit temporary GPU input allocations. The unique output matrix and the complete gathered output matrix still occupy GPU memory; this is not an unbounded streaming implementation. The computation cache is scoped to one call so changing the operation cannot reuse stale results. This initial implementation shares computation across a batch of versions, not a persistent on-disk GPU cache.

The GPU benchmark uses float32 row-wise real FFT: a concrete block-local spectral-analysis operation over the complete corpus bytes. Dense and reused paths have identical boundaries, padding, precision, block width and batch cap. This is not whole-file FFT, a trained model, a language-model inference benchmark or global attention. Global-context computations cannot safely reuse outputs based only on local byte equality.

Both paths reconstruct all original bytes, verify hashes, and return every GPU output to host memory. Full computed outputs must compare bitwise equal. A rounding difference between CUDA batching configurations fails the exact-reuse gate; it is not hidden behind a numerical tolerance.

## Measurements and review gate

The timer includes CPU boundary finding, hashing and lookup, packing, host-to-device transfer, FFT computation, GPU gathering, full device-to-host transfer, complete byte reconstruction and integrity checks. Input disk reads and independent final assertions are outside timing. CUDA synchronization brackets the complete wall timer; CUDA events separately measure the cast-plus-FFT regions. Those device regions omit transfer and gather costs, which remain in total wall time.

Seven repetitions alternate method order after warmup. Results include GPU model, Torch/CUDA versions, host CPU time, complete wall time, device FFT time, peak allocated GPU memory, exact data coverage, computed/reused block counts, all runs and medians. A single original dataset and a three-version workload are reported separately. The two edited versions are constructed insertions/replacements of the real pinned corpora, not a recorded editing history.

The review gate requires bitwise-identical full outputs and lower median complete wall time **and** lower median device FFT time on both workloads. Fewer computed blocks alone does not pass. This is a first measurement gate, not automatic authorization to publish. Small timing differences should be rerun and assessed for stability. Electricity savings are not measured.

## Run on the desktop GPU

Use the existing CUDA-enabled Python environment. The helper does not install or replace Torch or GPU drivers.

```powershell
conda activate torch5070
powershell -ExecutionPolicy Bypass -File scripts/run_gpu.ps1
```

Alternatively, from the repository root:

```bash
python scripts/download_corpora.py data/corpora
python scripts/benchmark_gpu.py data/corpora --repeats 7
```

Read `results/lossless/gpu_compute.json`. A blocked or failed status does not establish GPU savings. Generated corpus inputs are ignored by Git. Avoid publishing until actual hardware results are reviewed. The script requires no access to personal files; it downloads and verifies the pinned public corpus inputs.

## Validation completed in this session

45 tests pass. Four new CPU reference tests verify exact original-byte reconstruction, full dense/reused output equality, reuse after insertion, distinct tail lengths, changed operation configurations, integrity rejection and output shape checks. The CUDA runner and CUDA timing branch compile but have not executed here. No GPU timing or speedup is inferred from the CPU reference tests.

## Handoff for a desktop-enabled Work tab

Continue the GPU reduced-compute task in `dustindwayne/structural-residual-lm`. Read this document and run `scripts/run_gpu.ps1` in the user's existing CUDA environment. Measure full data and complete reconstructed outputs. Inspect the hardware JSON, fix any failures, and reassess whether the measured scoped GPU claim is strong enough to make public. Keep the repo private until that review; making it public has not yet been authorized.
