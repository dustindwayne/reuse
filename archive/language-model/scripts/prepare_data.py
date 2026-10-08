import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from structural_residual_lm.data import prepare
p=argparse.ArgumentParser();p.add_argument("--input");p.add_argument("--output",default="data");p.add_argument("--seed",type=int,default=19)
if __name__=="__main__":
    a=p.parse_args();print(prepare(a.output,a.seed,a.input))
