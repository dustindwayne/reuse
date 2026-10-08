import json,random,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.adaptive import encode,decode
from structural_residual_lm.grid import encode as grid,decode as ungrid
from structural_residual_lm.lossless import Grammar
rng=random.Random(19)
cases={'changing_ranges':bytes(rng.randrange(16)+(240 if (i//256)%2 else 0) for i in range(65536)),
'frequent_outlier':bytes(255 if rng.random()<.65 else rng.randrange(120,128) for _ in range(65536)),
'alternating_extremes':bytes([0,255])*32768,'random':rng.randbytes(65536),'synthetic_text':Path('archive/language-model/data/train.txt').read_bytes()}
rows=[]
for name,raw in cases.items():
    row=dict(name=name,raw_bytes=len(raw),direct_zlib_bytes=len(zlib.compress(raw)),transforms=[])
    for label,enc,dec in [('dynamic',encode,decode),('grid',grid,ungrid),('grammar',lambda b:Grammar.build(b).dumps(),lambda b:Grammar.loads(b).decode())]:
        t=time.perf_counter();transformed=enc(raw);packed=zlib.compress(transformed);elapsed=time.perf_counter()-t
        assert dec(zlib.decompress(packed))==raw
        row['transforms'].append(dict(name=label,transform_bytes=len(transformed),then_zlib_bytes=len(packed),encode_seconds=elapsed,exact_roundtrip=True))
    rows.append(row)
Path('results/lossless/composed.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
