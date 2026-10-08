"""Small lossy orthographic representation. No token identity lookup."""
from dataclasses import dataclass
import json
import numpy as np

@dataclass(frozen=True)
class StructuralEncoder:
    prefixes: tuple = ("re", "un", "pre")
    suffixes: tuple = ("ing", "ed", "ly", "tion", "s")
    chunks: tuple = ("th", "sh", "ch", "ng")
    disabled: tuple = ()

    @property
    def names(self):
        return ("length", "vowel_ratio", "alpha", "digit", "punctuation", "repeat",
                "initial_vowel", "final_vowel") + tuple("vc_"+str(i) for i in range(4)) + tuple("prefix_"+s for s in self.prefixes) + tuple("suffix_"+s for s in self.suffixes) + tuple("chunk_"+s for s in self.chunks)

    def encode(self, token):
        t = token.lower()
        n = max(len(t), 1)
        vowels = set("aeiou")
        features = [min(len(t), 16)/16, sum(c in vowels for c in t)/n,
                    float(t.isalpha()), float(t.isdigit()), float(bool(t) and all(not c.isalnum() for c in t)),
                    float(any(a == b for a,b in zip(t,t[1:]))),
                    float(bool(t) and t[0] in vowels), float(bool(t) and t[-1] in vowels)]
        features += [float(i < len(t) and t[i] in vowels) for i in range(4)]
        features += [float(t.startswith(s)) for s in self.prefixes]
        features += [float(t.endswith(s)) for s in self.suffixes]
        features += [float(s in t) for s in self.chunks]
        a = np.asarray(features, dtype=np.float64)
        for i,name in enumerate(self.names):
            if any(name.startswith(group+"_") for group in self.disabled): a[i] = 0
        return a

    def state(self, token):
        # Quantization enables counted state distributions, not exact word prediction.
        return tuple(np.rint(self.encode(token)*16).astype(int))

    def dumps(self):
        return json.dumps(dict(prefixes=self.prefixes, suffixes=self.suffixes, chunks=self.chunks, disabled=self.disabled), sort_keys=True)

    @classmethod
    def loads(cls, data):
        return cls(**{k:tuple(v) for k,v in json.loads(data).items()})
