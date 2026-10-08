import hashlib
import numpy as np
from .predictor import StructuralPredictor
from .surprise import surprise
from .gating import compute_levels

def feature_split(p,sequences,locations,config):
    base,signals=p.features(sequences,locations)
    scores=np.asarray([surprise(s["confidence"],s["entropy"],s["mismatch"],not config.get("no_confidence",False)) for s in signals])
    if config.get("random_surprise"):
        # Causal random allocation: hash only observed history, never future score arrays.
        def random_score(doc,i):
            key=str(config["seed"])+":"+" ".join(sequences[doc][max(0,i-4):i])
            return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")/2**64
        scores=np.asarray([random_score(doc,i) for doc,i in locations])
    lev=compute_levels(scores,tuple(config.get("thresholds",[.25,.50]))) if config["system"]=="residual" else np.full(len(signals),2)
    if "fixed_level" in config: lev=np.full(len(signals),config["fixed_level"])
    if config.get("no_gating"): lev=np.full(len(signals),2)
    if config["system"]=="residual":
        delta=np.asarray([s["residual"] for s in signals])
        if config.get("no_residual"): delta=np.zeros_like(delta)
        f=np.concatenate([base,delta],axis=1)
    else: f=base
    if config.get("no_confidence"): f[:,len(p.encoder.names)]=0
    if config.get("random_structure"):
        # Hash seed + observed token: independent of dataset order or future tokens.
        def random_feature(token):
            seed=int.from_bytes(hashlib.sha256((str(config["seed"])+":"+token).encode()).digest()[:8],"little")
            return np.random.default_rng(seed).uniform(0,1,f.shape[1])
        mapping={sequences[doc][i-1] for doc,i in locations}
        table={t:random_feature(t) for t in mapping}
        f=np.asarray([table[sequences[doc][i-1]] for doc,i in locations])
    log=dict(confidence=np.asarray([s["confidence"] for s in signals]),entropy=np.asarray([s["entropy"] for s in signals]),
             mismatch=np.asarray([s["mismatch"] for s in signals]),surprise=scores,compute_level=lev,
             context_depth=np.asarray([s["depth"] for s in signals]),support_count=np.asarray([s["count"] for s in signals]))
    return f,lev,log

def build_features(data,config):
    p=StructuralPredictor(config.get("trie_depth",3),config.get("disabled_features",()),config.get("no_trie",False)).fit(data["train_sequences"])
    arrays=[];levels=[];logs=[]
    for split in ["train","validation"]:
        f,lev,log=feature_split(p,data[split+"_sequences"],data[split][2],config)
        arrays.append(f);levels.append(lev);logs.append(log)
    return arrays,levels,logs,p
