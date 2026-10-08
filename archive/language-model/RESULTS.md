# Results

## Observed results

Designed grammar smoke corpus: 3,072 unique sentence documents, 80/20 document split, 26,512 training and 6,640 validation next-token examples. Validation sentences are disjoint, but lexical items and grammar templates overlap heavily. CPU NumPy runs, 250 Adam steps, batch 128, context 4, embedding 12, hidden 48, learning rate 0.005. These are implementation checks, not a natural-language evaluation.

Seed 7. Throughput is predictions per second in teacher-forced batched evaluation, not autoregressive generation tokens/sec. End-to-end includes structural extraction, surprise scoring, routing and neural evaluation; one-time trie fitting and disk reads are excluded. Median of three end-to-end or five neural-only repeats.

| System | Validation NLL | Accuracy | Neural-only predictions/sec | End-to-end predictions/sec | Process peak RAM MiB | Optional block work skipped | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.781270 | 0.6131 | 621060 | 663075 | 37.23 | 0.00% | 9984 |
| structural | 0.772843 | 0.6044 | 705202 | 7152 | 81.63 | 0.00% | 11232 |
| residual | 0.772936 | 0.6063 | 742786 | 7156 | 95.88 | 35.20% | 12384 |

VRAM is not applicable: these runs allocate no GPU tensors. RAM is Linux process high-water RSS, including training, dataset and prior construction; it is not the model's resident tensor size.

Skipped work is instrumented conditional block-row execution. The percentage counts only two optional H×H blocks; it does not count embeddings, output projection, routing or feature extraction. `estimated_dense_matmul_skipped` includes all dense matrix products in a forward pass and remains an arithmetic estimate, not measured FLOPs, energy or wall-time savings.

## Ablations across seeds

Seeds 7, 11, 23. Mean ± sample standard deviation across runs; three seeds are not a convincing statistical significance test. Conditions retain identical train/validation splits, tokenizer, steps, optimizer and batch sampling seed. Feature projections add parameters, so these are not exactly parameter-matched neural controls.

| Condition | Validation NLL mean ± SD | End-to-end predictions/sec mean | Block work skipped mean |
| --- | ---: | ---: | ---: |
| baseline | 0.774179 ± 0.006828 | 593373 | 0.00% |
| structural | 0.774217 ± 0.001532 | 7397 | 0.00% |
| residual | 0.773712 ± 0.002603 | 7067 | 35.20% |
| no_trie | 0.775735 ± 0.006214 | 7188 | 0.00% |
| no_prefix | 0.773013 ± 0.002867 | 6816 | 32.27% |
| no_suffix | 0.773625 ± 0.002228 | 6062 | 35.20% |
| no_chunk | 0.774595 ± 0.003143 | 6643 | 35.64% |
| no_vc | 0.777139 ± 0.004282 | 6592 | 33.37% |
| no_residual | 0.773450 ± 0.001935 | 7311 | 35.20% |
| no_gating | 0.773775 ± 0.001285 | 6955 | 0.00% |
| no_confidence | 0.775532 ± 0.003580 | 7470 | 39.80% |
| random_structure | 0.775905 ± 0.004070 | 7436 | 35.20% |
| random_surprise | 0.784131 ± 0.000972 | 7357 | 39.40% |
| fixed_compute | 0.776195 ± 0.005443 | 7142 | 50.00% |

Raw metrics/configs are under [results/ablations](../../results/ablations); per-example component CSVs preserve confidence, entropy, mismatch, surprise, depth, support and actual selected level. Historical classifier metrics are separate in [provenance](../../docs/PROVENANCE.md).

## Interpretation

The prototype demonstrates deterministic features, context-trie expectations, signed residual representation and genuinely conditional neural blocks. It does not demonstrate an end-to-end speedup. The structural path is much slower than the baseline once Python feature extraction is counted. Small loss differences should not be advertised as superiority; seed variation and random-feature controls must be considered.

Random-token features also perform close to the structural systems, weakening the claim that this feature design adds useful signal on the smoke corpus.

## Speculation

Caching representations, vectorizing trie prediction, or moving routing into a compiled implementation might reduce overhead. None was tested here, and savings cannot be inferred from skipped neural blocks alone. Larger natural-language corpora, capacity-matched controls, blind held-out thresholds and longer-context models are required before broader claims.
