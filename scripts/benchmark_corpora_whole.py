"""Whole-file transform composition, separated from chunk-container overhead."""
import hashlib,json,statistics,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import select,restore,pack,unpack
root=Path(sys.argv[1]);manifest=json.loads((Path(__file__).resolve().parents[1]/'results/lossless/corpus_sources.json').read_text())
out=Path(sys.argv[2]);report={'files':[],'matched_archives':[],'grammar_max_merges':32,'repeats':1}
def direct(raw):return min(((0,raw),(1,zlib.compress(raw))),key=lambda x:len(x[1]))
for item in manifest['files']:
 raw=(root/item['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 t=time.perf_counter();mode,payload=select(raw);enc=time.perf_counter()-t
 t=time.perf_counter();decoded=restore(mode,payload);dec=time.perf_counter()-t;assert decoded==raw
 report['files'].append(dict(path=item['path'],raw_bytes=len(raw),selected_mode=mode,selected_payload_bytes=len(payload),
                            direct_zlib_bytes=len(zlib.compress(raw)),encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True))
 out.write_text(json.dumps(report,indent=2)+'\n');print(item['path'],mode,len(payload),flush=True)
for corpus in ('canterbury','calgary'):
 files={Path(e['path']).name:(root/e['path']).read_bytes() for e in manifest['files'] if e['path'].startswith(corpus+'/')}
 t=time.perf_counter();blob=pack(files,chunk_size=32768,selector=direct);enc=time.perf_counter()-t
 t=time.perf_counter();assert unpack(blob)==files;dec=time.perf_counter()-t
 report['matched_archives'].append(dict(corpus=corpus,chunk_size=32768,bytes=len(blob),encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True))
out.write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',flush=True)
