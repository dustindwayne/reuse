import hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.patterns import pack,unpack
from structural_residual_lm.container import pack as fixed,unpack as unfixed
rng=random.Random(19);base=rng.randbytes(32768)
scenarios={'shifted_and_edited':{'base':base,'shifted':b'inserted prefix'+base,'edited':base[:10000]+b'EDIT'+base[10000:]}}
root=Path(sys.argv[1]);sources=json.loads(Path('results/lossless/corpus_sources.json').read_text())
for corpus in ('canterbury','calgary'):
 files={Path(e['path']).name:(root/e['path']).read_bytes() for e in sources['files'] if e['path'].startswith(corpus+'/')}
 for e in sources['files']:
  if e['path'].startswith(corpus+'/'):assert hashlib.sha256(files[Path(e['path']).name]).hexdigest()==e['sha256']
 scenarios[corpus]=files
rows=[]
for name,files in scenarios.items():
 row=dict(name=name,input_bytes=sum(map(len,files.values())),methods={})
 for label,encoder,decoder in [('pattern_first',pack,unpack),('same_pipeline_without_matches',lambda f:pack(f,matching=False),unpack)]:
  t=time.perf_counter();blob=encoder(files);enc=time.perf_counter()-t
  t=time.perf_counter();restored,stats=decoder(blob,True);dec=time.perf_counter()-t;assert restored==files
  row['methods'][label]=dict(bytes=len(blob),encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True,**stats)
 if name=='shifted_and_edited':
  t=time.perf_counter();blob=fixed(files);enc=time.perf_counter()-t;assert unfixed(blob)==files
  row['methods']['fixed_chunks']=dict(bytes=len(blob),encode_seconds=enc,exact_roundtrip=True)
 rows.append(row);Path('results/lossless/patterns.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
