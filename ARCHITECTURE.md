# Architecture

## Reasoning transition

The historical notebook reports Dustin asking:

> what if the submarine window expanded and contracted depending on needed context?

This is secondary evidence, not a recovered original message. This experiment narrows the question to whether predictable orthographic states can guide variable neural depth. It does not implement oversized-model streaming or a cognitive runtime.

## Structural representation

24 float features: capped length, vowel ratio, alphabetic/digit/punctuation flags, repetition, initial/final vowels, four leading vowel/consonant positions, three prefix flags, five suffix flags and four chunk flags. Numeric values lie in [0,1]. Quantized 1/16 states supply trie keys. This is lossy: different words can share a state. Morphology is heuristic; an `s` suffix flag is not a linguistic parser. Unicode words receive simple deterministic handling, not multilingual linguistic correctness.

## Causal flow

At position t the model has observed tokens through t and predicts t+1.

1. A train-only frozen reversed context trie predicts the expected state of t from structural states before t.
2. The observed state of t minus that expectation forms a signed numeric residual. It describes an observed token, never the unknown target t+1.
3. The same trie predicts the expected state of t+1 from history through t.
4. Neural inputs are the four observed token embeddings plus the expected next structural state. The residual configuration also supplies the residual of t.
5. Surprise of t selects zero, one or two optional neural blocks for predicting t+1.

| Configuration | Inputs beyond token embeddings | Optional blocks |
| --- | --- | --- |
| Baseline | None | Always two |
| Structural | Expected next state, prior confidence and entropy of observed t | Always two |
| Residual | Structural inputs plus signed observed-t residual | Conditional |

This tests lagged observed surprise as a predictor of required next-token computation. A system that needs the unknown target to compute surprise would leak the answer.

## Trie

Each node counts target structural states for a reversed suffix of preceding states. Prediction follows the longest observed suffix up to depth three, then backs off. State expectation is the count-weighted mean. Empirical support entropy is normalized by log(number of distinct states); singleton support has entropy zero. Confidence is maximum empirical probability times n/(n+2), a fixed support discount. Empty prior returns zero expectation, zero confidence and entropy one.

The trie is fit only on training documents. Document boundaries are retained. It is frozen for validation and inference. Training features use an offline prior fit to all training documents; this is a supervised train-only preprocessing choice, not an online prequential experiment.

## Residual and surprise

`r_t = encode(token_t) - E[state_t | past_t]`.

`mismatch_t = mean(abs(r_t))`.

`surprise_t = ((1-confidence_t) + entropy_t + mismatch_t)/3`.

All three terms are bounded [0,1]. Equal weighting, support discount and thresholds are explicit experimental choices, not theoretical guarantees or proven calibrated uncertainty. Components are logged separately. Scores below 0.25 use zero optional blocks, [0.25,0.50) use one, and ≥0.50 use two. Thresholds are fixed before evaluation.

## Neural model and execution

Concatenated token embeddings feed a tanh hidden projection. Structural features add a separate learned projection. Two optional residual blocks each compute `h += tanh(h W + b)`; only selected row indices enter their matrix multiplication. All rows share an output softmax. This is a fixed-context feed-forward LM, not a Transformer.

Manual reverse-mode gradients are verified numerically, including shared embedding indices and conditionally selected blocks. Adam uses β1=0.9, β2=0.999, epsilon=1e-8 and global norm clipping at 5. Shared tensor initialization and sampled batches match across systems for a given seed. Parameter counts differ because feature projections add tensors; this limitation is reported.

## Measurement boundaries

Neural-only timing uses precomputed features and excludes structural overhead. End-to-end timing recomputes causal structural features for every validation example. Both are batched teacher-forced throughput, not serving latency or autoregressive generation throughput. Checkpoint predictions rebuild the frozen training trie and verify corpus hashes.
