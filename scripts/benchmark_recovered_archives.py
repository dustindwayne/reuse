import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.patterns import pack,unpack
from structural_residual_lm.container import select
root=Path(sys.argv[1]);sources=json.loads(Path('results/lossless/corpus_sources.json').read_text());rows=[]
for corpus in ('canterbury','calgary'):
 files={Path(e['path']).name:(root/e['path']).read_bytes() for e in sources['files'] if e['path'].startswith(corpus+'/')}
 for matching in (False,True):
  for folding in (False,True):
   t=time.perf_counter();blob=pack(files,matching=matching,selector=lambda raw:select(raw,folding=folding));enc=time.perf_counter()-t
   t=time.perf_counter();restored,stats=unpack(blob,True);dec=time.perf_counter()-t;assert restored==files
   rows.append(dict(corpus=corpus,matching=matching,folding=folding,archive_bytes=len(blob),encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True,**stats))
   Path('results/lossless/recovered_archives.json').write_text(json.dumps(rows,indent=2)+'\n');print(corpus,matching,folding,len(blob),flush=True)
