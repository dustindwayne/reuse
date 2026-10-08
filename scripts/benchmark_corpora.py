"""Complete pinned Canterbury/Calgary evaluation; checkpoint after each file."""
import argparse,bz2,gzip,hashlib,io,json,lzma,platform,resource,statistics,sys,time,zipfile,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import pack,unpack

p=argparse.ArgumentParser();p.add_argument('--corpus-root',required=True);p.add_argument('--repeats',type=int,default=3)
p.add_argument('--output',default='results/lossless/corpora.json');a=p.parse_args()
root=Path(a.corpus_root);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)

def direct(raw):return min(((0,raw),(1,zlib.compress(raw))),key=lambda x:len(x[1]))

def timed(enc,dec,source,repeats):
    enc_times=[];dec_times=[];packed=None;stats=None
    for _ in range(repeats):
        t=time.perf_counter();blob=enc(source);enc_times.append(time.perf_counter()-t)
        if packed is not None:assert blob==packed
        packed=blob;t=time.perf_counter();restored=dec(blob);dec_times.append(time.perf_counter()-t)
        assert restored==source
    return dict(bytes=len(packed),encode_seconds=statistics.median(enc_times),decode_seconds=statistics.median(dec_times),
                encode_repeats=enc_times,decode_repeats=dec_times,exact_roundtrip=True),packed

report=dict(source_repository='https://github.com/pfalcon/canterbury-corpus',
            source_tree='c380ef9dc1dc62691757fdfeebdc2f85710e1e46',
            environment=dict(python=sys.version,platform=platform.platform(),zlib=zlib.ZLIB_RUNTIME_VERSION),
            repeats=a.repeats,files=[],archives=[])
algorithm_files=('container.py','adaptive.py','grid.py','lossless.py')
report['algorithm_sha256']={name:hashlib.sha256((Path(__file__).resolve().parents[1]/'src/structural_residual_lm'/name).read_bytes()).hexdigest() for name in algorithm_files}
report['parameters']={'outer_chunk_sizes':[4096,32768],'grammar_merges':32,'dynamic_max_window':1024,'dynamic_min_split':16,'zlib_level':6}

if out.exists():
    previous=json.loads(out.read_text())
    if previous.get("repeats")==a.repeats and previous.get("source_tree")==report["source_tree"] and previous.get("algorithm_sha256")==report["algorithm_sha256"]:
        report["files"]=previous["files"]

def save():
    report['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.write_text(json.dumps(report,indent=2)+'\n')

for corpus in ('canterbury','calgary'):
    manifest=json.loads((Path(__file__).resolve().parents[1]/'results/lossless/corpus_sources.json').read_text())
    paths=sorted(root/e['path'] for e in manifest['files'] if e['path'].startswith(corpus+'/'))
    for path in paths:
        expected=next(e for e in manifest['files'] if root/e['path']==path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']
    files={path.name:path.read_bytes() for path in paths}
    for name,raw in files.items():
        old=next((e for e in report["files"] if e["corpus"]==corpus and e["name"]==name),None)
        if old and old["sha256"]==hashlib.sha256(raw).hexdigest(): continue
        if old: report["files"].remove(old)
        entry=dict(corpus=corpus,name=name,input_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),methods={})
        source={name:raw}
        r,blob=timed(lambda f:pack(f),unpack,source,a.repeats);_,stats=unpack(blob,True);r.update(stats)
        entry['methods']['framework_4k']=r
        r,_=timed(lambda f:pack(f,selector=direct),unpack,source,a.repeats);entry['methods']['matched_zlib_4k']=r
        for label,enc,dec in [('zlib',zlib.compress,zlib.decompress),('gzip',lambda b:gzip.compress(b,mtime=0),gzip.decompress),
                              ('bzip2',bz2.compress,bz2.decompress),('xz',lzma.compress,lzma.decompress)]:
            r,_=timed(enc,dec,raw,a.repeats);entry['methods'][label]=r
        report['files'].append(entry);save()
        print(json.dumps({'file':corpus+'/'+name,'framework':entry['methods']['framework_4k']['bytes'],
                         'matched':entry['methods']['matched_zlib_4k']['bytes'],'zlib':entry['methods']['zlib']['bytes']}),flush=True)
    for size in (4096,32768):
        r,blob=timed(lambda f:pack(f,chunk_size=size),unpack,files,1);_,stats=unpack(blob,True);r.update(stats)
        report['archives'].append(dict(corpus=corpus,method='framework',chunk_size=size,input_bytes=sum(map(len,files.values())),**r));save()
    r,_=timed(lambda f:pack(f,selector=direct),unpack,files,a.repeats)
    report['archives'].append(dict(corpus=corpus,method='matched_zlib',chunk_size=4096,**r))
    def zipped(f):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in f.items():
                info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
        return buffer.getvalue()
    def unzip(blob):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:return {n:z.read(n) for n in z.namelist()}
    r,_=timed(zipped,unzip,files,a.repeats);report['archives'].append(dict(corpus=corpus,method='zip',**r));save()
print('COMPLETE',flush=True)
