"""Full-input, complete encode/restore CPU measurements with matched caching."""
import argparse,hashlib,json,platform,statistics,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.fast import Store,loads

def run(versions,options):
    start=time.process_time();wall=time.perf_counter();store=Store(**options);stages=[]
    for name,files in versions.items():
        before=time.process_time();store.add(name,files);stages.append(time.process_time()-before)
    blob=store.dumps();enc=time.process_time()-start;encwall=time.perf_counter()-wall
    start=time.process_time();wall=time.perf_counter();restored=loads(blob)
    dec=time.process_time()-start;decwall=time.perf_counter()-wall
    assert restored==versions
    return dict(bytes=len(blob),encode_cpu_s=enc,decode_cpu_s=dec,total_cpu_s=enc+dec,total_wall_s=encwall+decwall,first_add_cpu_s=stages[0],updates_cpu_s=sum(stages[1:]),**store.stats)

def run_direct(versions):
    start=time.process_time();wall=time.perf_counter()
    encoded={v:{p:zlib.compress(raw,6) for p,raw in files.items()} for v,files in versions.items()}
    enc=time.process_time()-start;encwall=time.perf_counter()-wall
    start=time.process_time();wall=time.perf_counter()
    restored={v:{p:zlib.decompress(raw) for p,raw in files.items()} for v,files in encoded.items()}
    dec=time.process_time()-start;decwall=time.perf_counter()-wall
    assert restored==versions
    return dict(bytes=sum(len(raw) for files in encoded.values() for raw in files.values()),encode_cpu_s=enc,decode_cpu_s=dec,total_cpu_s=enc+dec,total_wall_s=encwall+decwall)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('corpora',type=Path);parser.add_argument('--repeats',type=int,default=7);parser.add_argument('--output',type=Path,default=Path('results/lossless/fast_compute.json'));args=parser.parse_args()
    manifest=json.loads(Path('results/lossless/corpus_sources.json').read_text());original={}
    for entry in manifest['files']:
        raw=(args.corpora/entry['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];original[entry['path']]=raw
    versions={'original':original};changed={};second={}
    for name,raw in original.items():
        at=len(raw)//3;edited=raw[:at]+b'\nlocal inserted record\n'+raw[at:];changed[name]=edited
        at=2*len(edited)//3;second[name]=edited[:at]+b'updated record'+edited[at+17:]
    versions['insert']=changed;versions['replace']=second
    methods={'whole_file_cached':dict(whole_file=True,adaptive=False),'fixed_cached':dict(content=False,adaptive=False),'content_cached':dict(adaptive=False),'content_sampled':dict(adaptive=True)}
    # Warm up all paths, then rotate execution order to limit order bias.
    for options in methods.values():run(versions,options)
    samples={name:[] for name in methods};cold={name:[] for name in methods};samples['direct_codec']=[];cold['direct_codec']=[]
    for repeat in range(args.repeats):
        samples['direct_codec'].append(run_direct(versions));cold['direct_codec'].append(run_direct({'original':original}))
        names=list(methods);names=names[repeat%len(names):]+names[:repeat%len(names)]
        for name in names:
            samples[name].append(run(versions,methods[name]));cold[name].append(run({'original':original},methods[name]))
    result={'python':platform.python_version(),'platform':platform.platform(),'zlib':zlib.ZLIB_RUNTIME_VERSION,'repeats':args.repeats,'level':6,'source_files':len(original),'source_bytes':sum(map(len,original.values())),'version_count':len(versions),'processed_bytes':sum(len(raw) for files in versions.values() for raw in files.values()),'workload':'Complete pinned Canterbury and Calgary files, then constructed insertion at one-third and replacement at two-thirds of every file. All versions restored and verified.','methods':{name:{'median':{key:statistics.median(row[key] for row in rows) for key in rows[0]},'runs':rows,'single_version_median':{key:statistics.median(row[key] for row in cold[name]) for key in cold[name][0]},'single_version_runs':cold[name]} for name,rows in samples.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:row['median'] for name,row in result['methods'].items()},indent=2))
if __name__=='__main__':main()
