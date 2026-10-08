# Experiment trail

The current objective is less **complete elapsed time**, preserving full data and outputs. Earlier attempts remain available to show what changed and why.

| Stage | Evidence | Outcome |
| --- | --- | --- |
| Current GPU repeatability | [Frozen lifecycle report](GPU_LIFECYCLE.md) | Four/eight constructed edits favorable across three sessions; two edits unreliable; tested code, raw data and freeze integrated |
| First lifecycle measurement | [Initial lifecycle archive](archive/GPU_LIFECYCLE_INITIAL.md) | First favorable two-edit point later corrected |
| Chunk-level GPU cache | [Chunk lookup archive](archive/GPU_CHUNK_LOOKUP.md) | Faster updates, slower startup; update-only gains insufficient |
| Optimized dense / block cache | [Block lookup archive](archive/GPU_LOCAL_OPTIMIZATION.md) | Correcting dense baseline erased earlier reuse advantage |
| Checked-in GPU prototype | [Prototype archive](archive/GPU_PROTOTYPE.md) | Initial implementation; workspace GPU unavailable; predates local tuning |
| CPU complete-data path | [Fast compute](FAST_COMPUTE.md) | Measured encode-plus-restore CPU reduction; single-version storage tradeoff |
| Reversible lane folding | [Recovered integration](RECOVERED_INTEGRATION.md) | Some size gains; exhaustive selection too expensive for compute objective |
| Pattern matching | [Pattern-first](PATTERN_FIRST.md) | Shifted reuse worked; greedy matching could enlarge real archives |
| Shared containers / transforms | [Container](CONTAINER.md), [adaptive](ADAPTIVE.md), [grid](GRID.md), [composition](COMPOSED.md), [lossless](LOSSLESS.md) | Lossless representation experiments; size gains alone do not establish compute gains |
| Real corpus inputs | [Corpus evaluation](CORPUS_EVALUATION.md) | Complete pinned Canterbury/Calgary evaluation |
| Earlier language model | [Historical overview](archive/EARLIER_LANGUAGE_MODEL.md), [results](../archive/language-model/RESULTS.md), [architecture](../archive/language-model/ARCHITECTURE.md) | Optional neural work skipped; structural overhead prevented overall speedup |

[Provenance](PROVENANCE.md), [failures](FAILURE_LOG.md), [timeline](TIMELINE.md), and Git history retain attribution and corrections. Archived reports reflect knowledge at their stage; use the current lifecycle report for GPU conclusions. Historical language-model résumé wording is not the current framework description.
