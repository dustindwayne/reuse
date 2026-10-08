import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]: os.environ.setdefault(key,"1")
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from structural_residual_lm.data import load,TOKEN_PATTERN
from structural_residual_lm.model import NeuralLM
from structural_residual_lm.predictor import StructuralPredictor
from structural_residual_lm.experiment import feature_split

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="results/checkpoints/residual");p.add_argument("--text",default="the frog");a=p.parse_args()
    meta=json.loads(Path(a.checkpoint+".json").read_text());cfg=meta["config"];vocab=meta["vocab"];weights=np.load(a.checkpoint+".npz")
    m=NeuralLM(len(vocab),cfg["context"],cfg["embedding"],cfg["hidden"],features=weights["F"].shape[0] if "F" in weights else 0,seed=cfg["seed"])
    m.p={k:weights[k] for k in weights.files};tokens=["<bos>"]+TOKEN_PATTERN.findall(a.text.lower());ids={t:i for i,t in enumerate(vocab)}
    context=[ids["<bos>"]]*max(0,cfg["context"]-len(tokens))+[ids.get(t,0) for t in tokens[-cfg["context"]:]]
    f=levels=None
    if cfg["system"]!="baseline":
        d=load(cfg.get("data","data"),cfg["context"])
        if d["hashes"]!=meta["dataset_hashes"]:raise ValueError("Checkpoint corpus hashes differ")
        prior=StructuralPredictor(cfg.get("trie_depth",3),cfg.get("disabled_features",()),cfg.get("no_trie",False)).fit(d["train_sequences"])
        f,levels,_=feature_split(prior,[tokens],[(0,len(tokens))],cfg)
    prob,_=m.forward(np.asarray([context]),f,levels)
    print(json.dumps([dict(token=vocab[i],probability=float(prob[0,i])) for i in np.argsort(prob[0])[-5:][::-1]],indent=2))
if __name__=="__main__":main()
