"""Reversed context trie with longest observed suffix backoff."""
from collections import Counter
from dataclasses import dataclass,field
import numpy as np

@dataclass
class Node:
    counts: Counter = field(default_factory=Counter)
    children: dict = field(default_factory=dict)

class ContextTrie:
    def __init__(self, depth=3):
        if depth < 0: raise ValueError("depth must be nonnegative")
        self.depth=depth; self.root=Node()

    def update(self, context, state):
        node=self.root; node.counts[tuple(state)]+=1
        for key in reversed(context[-self.depth:] if self.depth else []):
            node=node.children.setdefault(tuple(key),Node()); node.counts[tuple(state)]+=1

    def predict(self, context, dimension):
        node=self.root; matched=0
        for key in reversed(context[-self.depth:] if self.depth else []):
            child=node.children.get(tuple(key))
            if child is None: break
            node=child; matched+=1
        total=sum(node.counts.values())
        if not total:
            return dict(expected=np.zeros(dimension), confidence=0., entropy=1., depth=0, count=0)
        probabilities=np.asarray(list(node.counts.values()),dtype=float)/total
        states=np.asarray(list(node.counts),dtype=float)/16
        entropy=float(-(probabilities*np.log(probabilities)).sum())
        # Normalize against support; singleton support has zero empirical entropy.
        entropy=entropy/np.log(len(probabilities)) if len(probabilities)>1 else 0.
        # Discount confidence in singleton/rare contexts by count/(count+2).
        return dict(expected=probabilities@states, confidence=float(probabilities.max()*total/(total+2)),
                    entropy=entropy, depth=matched, count=total)

    def fit(self, sequences):
        for seq in sequences:
            for i,s in enumerate(seq): self.update(seq[max(0,i-self.depth):i],s)
        return self

    def node_count(self):
        def count(n): return 1+sum(count(c) for c in n.children.values())
        return count(self.root)
