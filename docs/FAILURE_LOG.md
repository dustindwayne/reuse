# Failure log

## Structural overhead removes the practical speedup

### Hypothesis

Structural predictability can skip neural work without unacceptable prediction degradation.

### Expected result

Similar held-out loss and improved total prediction throughput.

### Observed result

Conditional blocks skip substantial row work, but end-to-end structural prediction is far slower than the small neural-only baseline. The first timing boundary omitted feature extraction and therefore overstated usable throughput.

### Evidence

[RESULTS.md](../RESULTS.md) records both timing boundaries. The initial neural-only record remains in Git history, before the timing correction commit.

### Diagnosis

Per-example Python token feature encoding, trie traversal and score assembly dominate the small NumPy neural model. Sparse neural matmuls do not remove that cost.

### Revision

Add end-to-end timing, preserve neural-only timing under an explicit name, and count actual optional block row calls separately from arithmetic estimates.

### Retest

All three configurations rerun with the corrected boundary; no end-to-end speedup demonstrated. This is an experimental negative result, not a completed performance optimization.

## Historical routing and cache-reuse limits — reported, not rerun

### Hypothesis

Selective loading reduces active memory without losing quality or throughput.

### Observed result

The historical top-2 classifier report nearly preserves global macro F1 and reduces active size, while inference time increases. The continuation separately reports zero RAM/VRAM cache hits in later generation.

### Evidence

[PROVENANCE.md](PROVENANCE.md) identifies the exact retrieved documents. Original logs and datasets were not recovered.

### Diagnosis

Routing overhead and ineffective cross-token cache reuse are plausible explanations, not established causes.

### Revision / retest

No historical rerun attempted. These reports justify distinguishing successful execution from hypothesis confirmation.

## Randomized routing and incomplete vowel removal

### Hypothesis / expected result

A negative control must not depend on future tokens, and a named feature removal must remove the advertised family.

### Observed result

Initial randomized-surprise control permuted the whole evaluation score array. Although it did not read labels directly, it used future-context statistics and was not a causal deployed route. Initial vowel removal zeroed only position bits while retaining vowel ratio and initial/final flags.

### Evidence / diagnosis

The initial controls remain in Git history. Source review identified both mismatches; a future-token mutation test now covers the randomized control.

### Revision

Use a stable hash of observed context for random surprise, and zero all vowel-related flags for the vowel ablation. Uniform random scores do not exactly match the adaptive compute budget; report that limit.

### Retest

Rerun both affected conditions at seeds 7, 11 and 23. Final results use only corrected values; old controls are not evidence for the final conclusions.
