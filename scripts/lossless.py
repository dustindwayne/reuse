"""Pack or exactly reconstruct a local file using the shared byte grammar."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.lossless import Grammar
p=argparse.ArgumentParser()
p.add_argument('operation',choices=['pack','unpack'])
p.add_argument('input');p.add_argument('output')
a=p.parse_args();raw=Path(a.input).read_bytes()
out=Grammar.build(raw).dumps() if a.operation=='pack' else Grammar.loads(raw).decode()
# Never silently replace a user's existing file.
with Path(a.output).open('xb') as f: f.write(out)
print(f'{len(raw)} -> {len(out)} bytes')
