"""Causal structural signals; frozen train-only prior and no target access."""
import numpy as np
from .encoder import StructuralEncoder
from .trie import ContextTrie
from .residual import residual

class StructuralPredictor:
    def __init__(self,depth=3,disabled=(),no_trie=False):
        self.encoder=StructuralEncoder(disabled=tuple(disabled)); self.trie=ContextTrie(depth);self.no_trie=no_trie

    def fit(self,sequences):
        self.trie.fit([[self.encoder.state(t) for t in seq] for seq in sequences]);return self

    def signal(self, history):
        if not history: raise ValueError("Need observed history")
        e=self.encoder;states=[e.state(t) for t in history];dimension=len(e.names)
        previous=self.trie.predict(states[:-1],dimension)
        next_state=self.trie.predict(states,dimension)
        if self.no_trie:
            for p in [previous,next_state]: p.update(expected=np.zeros(dimension),confidence=0.,entropy=1.,depth=0,count=0)
        observed=e.encode(history[-1]);delta=residual(observed,previous["expected"])
        mismatch=float(np.abs(delta).mean())
        return dict(prior=next_state["expected"],residual=delta,confidence=previous["confidence"],
                    entropy=previous["entropy"],mismatch=mismatch,depth=previous["depth"],count=previous["count"])

    def features(self,sequences,locations):
        rows=[];signals=[]
        for doc,i in locations:
            # i is the next target; history contains only tokens < i.
            signal=self.signal(sequences[doc][:i]);signals.append(signal)
            rows.append(np.r_[signal["prior"],signal["confidence"],signal["entropy"]])
        return np.asarray(rows),signals
