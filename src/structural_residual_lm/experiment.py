import numpy as np
from .predictor import StructuralPredictor
from .surprise import surprise
from .gating import compute_levels

def build_features(data,config):
    p=StructuralPredictor(config.get("trie_depth",3),config.get("disabled_features",()),config.get("no_trie",False)).fit(data["train_sequences"])
    arrays=[];logs=[];levels=[];rng=np.random.default_rng(config["seed"]+1000)
    random_map=None
    if config.get("random_structure"):
        # Stable random-token control of identical dimension, with no target access.
        random_map={t:rng.uniform(0,1,2*len(p.encoder.names)+2) for seq in data["train_sequences"]+data["validation_sequences"] for t in seq}
    for split in ["train","validation"]:
        seq=data[split+"_sequences"];locations=data[split][2];base,signals=p.features(seq,locations)
        scores=np.asarray([surprise(s["confidence"],s["entropy"],s["mismatch"],not config.get("no_confidence",False)) for s in signals])
        if config.get("random_surprise"): scores=rng.permutation(scores)
        lev=compute_levels(scores,tuple(config.get("thresholds",[.25,.50]))) if config["system"]=="residual" else np.full(len(signals),2)
        if "fixed_level" in config: lev=np.full(len(signals),config["fixed_level"])
        if config.get("no_gating"): lev=np.full(len(signals),2)
        if config["system"]=="residual":
            delta=np.asarray([s["residual"] for s in signals])
            if config.get("no_residual"): delta=np.zeros_like(delta)
            f=np.concatenate([base,delta],axis=1)
        else: f=base
        if config.get("no_confidence"): f[:,len(p.encoder.names)]=0
        if random_map is not None: f=np.asarray([random_map[seq[doc][i-1]][:f.shape[1]] for doc,i in locations])
        arrays.append(f);levels.append(lev)
        logs.append(dict(confidence=np.asarray([s["confidence"] for s in signals]),entropy=np.asarray([s["entropy"] for s in signals]),
                         mismatch=np.asarray([s["mismatch"] for s in signals]),surprise=scores,compute_level=lev,
                         context_depth=np.asarray([s["depth"] for s in signals]),support_count=np.asarray([s["count"] for s in signals])))
    return arrays,levels,logs,p
