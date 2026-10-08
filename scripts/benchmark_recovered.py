"""Test recovered lane transforms against the existing complete selector."""
import hashlib,json,sys,time,zlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import select,restore
from structural_residual_lm.folding import candidates
root=Path(sys.argv[1]);rows=[]
sources=json.loads(Path('results/lossless/corpus_sources.json').read_text());models=json.loads(Path('results/lossless/fold_model_sources.json').read_text())
inputs=[]
for e in sources['files']:inputs.append((e['path'],(root/e['path']).read_bytes(),'real_corpus',e['sha256']))
for e in models['files']:inputs.append(('models/'+e['local_name'],(root/'models'/e['local_name']).read_bytes(),'real_small_model',e['sha256']))
rng=np.random.default_rng(19);values=rng.normal(size=65536)
for name,raw in [('fp32_normal',values.astype('<f4').tobytes()),('fp16_normal',values.astype('<f2').tobytes()),
                 ('bf16_like',values.astype('<f4').view('<u4').__rshift__(16).astype('<u2').tobytes()),
                 ('random',rng.bytes(131072)),('already_compressed',zlib.compress(values.astype('<f4').tobytes()))]:
 inputs.append((name,raw,'synthetic_control',hashlib.sha256(raw).hexdigest()))
for name,raw,kind,digest in inputs:
 assert hashlib.sha256(raw).hexdigest()==digest
 t=time.perf_counter();old=select(raw,folding=False);old_seconds=time.perf_counter()-t;assert restore(*old)==raw
 t=time.perf_counter();folds=candidates(raw);fold_seconds=time.perf_counter()-t
 for mode,payload in folds:assert restore(mode,payload)==raw
 best_fold=min(folds,key=lambda x:len(x[1]));best=min([old]+folds,key=lambda x:len(x[1]))
 rows.append(dict(name=name,kind=kind,raw_bytes=len(raw),sha256=digest,direct_zlib_bytes=len(zlib.compress(raw)),
                 prior_framework_bytes=len(old[1]),new_framework_bytes=len(best[1]),selected_mode=best[0],
                 selected_width=best[1][0] if best[0]==8 else None,best_fold_bytes=len(best_fold[1]),
                 prior_search_seconds=old_seconds,additional_folding_seconds=fold_seconds,exact_roundtrip=True,
                 lane_trials=[dict(mode=m,width=p[0] if m==8 else None,bytes=len(p)) for m,p in folds]))
 Path('results/lossless/recovered_folding.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,len(old[1]),len(best[1]),best[0],best[1][0] if best[0]==8 else '',flush=True)
