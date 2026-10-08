from pathlib import Path
import yaml
from benchmark import run
if __name__=="__main__": run(yaml.safe_load(Path("configs/residual.yaml").read_text()))
