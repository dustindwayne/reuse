import hashlib,json,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import pack,unpack
from structural_residual_lm.framework import filewise,unpack as unpack_any
root=Path(sys.argv[1]);sources=json.loads(Path('results/lossless/corpus_sources.json').read_text());rows=[]
for corpus in ('canterbury','calgary'):
 files={Path(e['path']).name:(root/e['path']).read_bytes() for e in sources['files'] if e['path'].startswith(corpus+'/')}
 t=time.perf_counter();blob=filewise(files);enc=time.perf_counter()-t
 t=time.perf_counter();restored=unpack_any(blob);dec=time.perf_counter()-t;assert restored==files
 inner=zlib.decompress(blob[5:]) if blob[4] else blob[5:]
 rows.append(dict(corpus=corpus,archive_bytes=len(blob),inner_bytes=len(inner),outer_zlib=bool(blob[4]),encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True))
 Path('results/lossless/recovered_fileaware.json').write_text(json.dumps(rows,indent=2)+'\n');print(corpus,len(blob),flush=True)
