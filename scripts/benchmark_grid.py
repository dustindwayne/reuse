import hashlib,json,random,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.grid import encode,decode
rng=random.Random(19)
cases={'your_grid_decimal':bytes([1,11,2,20,0,22,33,21]),
       'your_grid_hex':bytes.fromhex('01 11 02 20 00 22 33 21'),
       'constant':b's'*65536,
       'clustered':bytes(rng.randrange(120,136) for _ in range(65536)),
       'smooth_grid':bytes((x//8+y//8)%256 for y in range(256) for x in range(256)),
       'random':rng.randbytes(65536),'synthetic_text':Path('archive/language-model/data/train.txt').read_bytes()}
rows=[]
for name,raw in cases.items():
    shapes=[(4,2,2)] if name.startswith('your_grid') else [(64,2,2),(64,8,8),(256,16,16),(256,32,32)]
    trials=[]
    for width,bw,bh in shapes:
        t=time.perf_counter(); average=encode(raw,width,bw,bh,False); adaptive=encode(raw,width,bw,bh,True); elapsed=time.perf_counter()-t
        assert decode(average)==raw and decode(adaptive)==raw
        trials.append(dict(width=width,block_width=bw,block_height=bh,average_bytes=len(average),adaptive_bytes=len(adaptive),two_encodes_seconds=elapsed))
    best=min(trials,key=lambda x:x['adaptive_bytes'])
    rows.append(dict(name=name,raw_bytes=len(raw),zlib_bytes=len(zlib.compress(raw)),best_adaptive_bytes=best['adaptive_bytes'],
                     sha256=hashlib.sha256(raw).hexdigest(),exact_roundtrip=True,trials=trials))
Path('results/lossless/grid.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('name','raw_bytes','zlib_bytes','best_adaptive_bytes')} for r in rows],indent=2))
