import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]: os.environ.setdefault(key,"1")
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from structural_residual_lm.data import load
from structural_residual_lm.model import NeuralLM
from structural_residual_lm.predictor import StructuralPredictor
from structural_residual_lm.metrics import evaluate,environment

def run(config,output=None):
    start=time.perf_counter();d=load(config.get("data","data"),config["context"]);x,y,_=d["train"];vx,vy,_=d["validation"]
    f=vf=None;predictor=None;feature_start=time.perf_counter()
    if config["system"]=="structural":
        predictor=StructuralPredictor(config.get("trie_depth",3)).fit(d["train_sequences"])
        f,signals=predictor.features(d["train_sequences"],d["train"][2]);vf,vsignals=predictor.features(d["validation_sequences"],d["validation"][2])
    elif config["system"]!="baseline": raise ValueError("Unknown system")
    feature_seconds=time.perf_counter()-feature_start
    m=NeuralLM(len(d["vocab"]),config["context"],config["embedding"],config["hidden"],features=0 if f is None else f.shape[1],seed=config["seed"])
    rng=np.random.default_rng(config["seed"]);curve=[];train_start=time.perf_counter()
    for step in range(config["steps"]):
        ids=rng.integers(len(y),size=config["batch_size"]);loss=m.train_batch(x[ids],y[ids],None if f is None else f[ids],lr=config["learning_rate"])
        if step%50==0 or step==config["steps"]-1: curve.append(dict(step=step+1,train_batch_loss=loss,validation_loss=evaluate(m,vx,vy,vf)["loss"]))
    train_seconds=time.perf_counter()-train_start
    validation=evaluate(m,vx,vy,vf);train=evaluate(m,x,y,f)
    evaluate(m,vx,vy,vf);latencies=[]
    for _ in range(5):
        t=time.perf_counter();evaluate(m,vx,vy,vf);latencies.append(time.perf_counter()-t)
    latency=float(np.median(latencies))
    result=dict(system=config["system"],config=config,timestamp=datetime.now(timezone.utc).isoformat(),environment=environment(),dataset_hashes=d["hashes"],
                parameters=m.parameter_count,structural_preprocessing_seconds=feature_seconds,trie_nodes=0 if predictor is None else predictor.trie.node_count(),training_examples=len(y),validation_examples=len(vy),train=train,validation=validation,
                train_seconds=train_seconds,training_tokens_per_second=config["steps"]*config["batch_size"]/train_seconds,
                neural_eval_seconds=latency,neural_eval_tokens_per_second=len(vy)/latency,neural_eval_repeats_seconds=latencies,
                estimated_block_compute_skipped=0.,curve=curve,runtime_seconds=time.perf_counter()-start)
    out=Path(output or f"results/{config['system']}.json");out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(out),"loss":validation["loss"],"tokens_per_second":len(vy)/latency}));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output");a=p.parse_args();run(yaml.safe_load(Path(a.config).read_text()),a.output)
