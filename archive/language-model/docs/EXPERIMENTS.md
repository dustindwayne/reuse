# Experiments

Git history records the implementation and experiment sequence.

1. Preserve source evidence and attribution gaps.
2. Test compact representation and counted context trie.
3. Train and commit neural baseline before adding structural integration.
4. Train and commit ungated structural model.
5. Add signed residual and conditional blocks; run first residual experiment.
6. Correct timing boundary to include structural extraction; rerun all systems.
7. Run component removals, random controls and fixed allocation across three seeds.

Dataset hash and full config accompany every result. Timing varies by CPU scheduling; NLL reproducibility is the primary deterministic check. Process RSS and latency can vary between reruns.

`no_trie` sets expectation/confidence to zero and entropy to one. `no_prefix`, `no_suffix`, `no_chunk` zero the indicated orthographic feature family. `no_vc` zeros the leading vowel/consonant position bits, vowel ratio and initial/final vowel flags. `no_residual` zeros the neural residual input but leaves mismatch available to the gate. `no_confidence` removes the neural confidence input and that term from surprise. `no_gating` uses both blocks. Random token features have stable hashed seeds and identical input dimension; random-surprise allocations hash the observed four-token history to a uniform score. They are causal and independent of actual surprise, but the resulting compute budget is not exactly matched to adaptive routing. Fixed compute uses one block (a different budget from adaptive inference).

The structural predictor was not tuned on validation. The smoke corpus intentionally repeats grammatical scaffolding; lexical generalization, novel grammar, natural text, semantic complexity and scale remain untested. Prefix/suffix removals can also change trie partitions, so they alter both structural inputs and the deterministic prior.
