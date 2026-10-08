import hashlib,json,random,sys,time,zipfile,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import pack,unpack
rng=random.Random(19);shared=rng.randbytes(16384)
scenarios={'mixed':{'text.txt':Path('archive/language-model/data/train.txt').read_bytes(),'pattern.bin':bytes([0,255])*8192,
                   'values.bin':bytes(255 if rng.random()<.65 else rng.randrange(120,128) for _ in range(16384)),'random.bin':rng.randbytes(16384)},
'shared_files':{'first.bin':shared*4,'second.bin':shared*3,'third.bin':shared+bytes([127])*4096},
'unique_random':{'random.bin':rng.randbytes(65536)}}
rows=[]
for name,files in scenarios.items():
    t=time.perf_counter();blob=pack(files);enc=time.perf_counter()-t
    t=time.perf_counter();decoded,stats=unpack(blob,True);dec=time.perf_counter()-t
    assert decoded==files
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for path,data in files.items():z.writestr(path,data)
    rows.append(dict(name=name,raw_bytes=sum(map(len,files.values())),container_bytes=len(blob),zip_bytes=len(buffer.getvalue()),
                encode_seconds=enc,decode_seconds=dec,exact_roundtrip=True,archive_sha256=hashlib.sha256(blob).hexdigest(),**stats))
Path('results/lossless/container.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
