"""Each condition runs in a fresh process for comparable process peak RSS."""
import argparse,json,subprocess,sys
from pathlib import Path
import yaml

CONDITIONS={
    "baseline":dict(system="baseline"),"structural":dict(system="structural"),"residual":{},
    "no_trie":dict(no_trie=True),"no_prefix":dict(disabled_features=["prefix"]),
    "no_suffix":dict(disabled_features=["suffix"]),"no_chunk":dict(disabled_features=["chunk"]),
    "no_vc":dict(disabled_features=["vc"]),"no_residual":dict(no_residual=True),
    "no_gating":dict(no_gating=True),"no_confidence":dict(no_confidence=True),
    "random_structure":dict(random_structure=True),"random_surprise":dict(random_surprise=True),
    "fixed_compute":dict(fixed_level=1)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",default="7,11,23");p.add_argument("--conditions",default=",".join(CONDITIONS));p.add_argument("--steps",type=int)
    a=p.parse_args();base=yaml.safe_load(Path("configs/residual.yaml").read_text());folder=Path("results/ablations");folder.mkdir(parents=True,exist_ok=True);summary=[]
    for name in a.conditions.split(","):
        override=CONDITIONS[name]
        for seed in map(int,a.seeds.split(",")):
            cfg=dict(base,**override);cfg["seed"]=seed
            if a.steps:cfg["steps"]=a.steps
            path=folder/f"{name}_{seed}.yaml";path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            output=folder/f"{name}_{seed}.json"
            subprocess.run([sys.executable,"scripts/benchmark.py","--config",str(path),"--output",str(output)],check=True)
            result=json.loads(output.read_text());summary.append(dict(condition=name,seed=seed,loss=result["validation"]["loss"],
                accuracy=result["validation"]["accuracy"],end_to_end_tokens_per_second=result["end_to_end_tokens_per_second"],
                neural_eval_tokens_per_second=result["neural_eval_tokens_per_second"],block_skipped=result["estimated_block_compute_skipped"],parameters=result["parameters"]))
    # Merge partial reruns without silently losing other completed conditions.
    merged={}
    for condition in CONDITIONS:
        for result_path in sorted(folder.glob(condition+"_[0-9]*.json")):
            result=json.loads(result_path.read_text());seed=result["config"]["seed"]
            merged[(condition,seed)]=dict(condition=condition,seed=seed,loss=result["validation"]["loss"],
                accuracy=result["validation"]["accuracy"],end_to_end_tokens_per_second=result["end_to_end_tokens_per_second"],
                neural_eval_tokens_per_second=result["neural_eval_tokens_per_second"],block_skipped=result["estimated_block_compute_skipped"],parameters=result["parameters"])
    (folder/"summary.json").write_text(json.dumps(list(merged.values()),indent=2)+"\n")
if __name__=="__main__":main()
