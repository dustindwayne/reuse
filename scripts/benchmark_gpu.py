"""CUDA-required, full-data block-local spectral processing benchmark.

No CPU fallback can produce a GPU-success result. FFT is a block-local data
analysis workload, not a trained model or a whole-file spectral transform.
"""
import argparse,hashlib,json,platform,statistics,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.gpu_reuse import BlockPlan,execute_torch


def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


def measure(torch,versions,reuse,width,batch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();cpu=time.process_time()
    plan=BlockPlan(width=width,reuse=reuse)
    for name,files in versions.items():plan.add(name,files)
    events=[]
    output=execute_torch(plan,lambda rows:torch.fft.rfft(rows.to(torch.float32),dim=1),
                         batch_blocks=batch,events=events)
    # Bring ALL results back, not just checksums or samples.
    restored_output=output.cpu().numpy()
    restored_data=plan.reconstruct()
    torch.cuda.synchronize()
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    assert restored_data==versions
    result=dict(total_wall_s=wall,host_cpu_s=cpu,
                fft_device_ms=sum(a.elapsed_time(b) for a,b in events),
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                output_bytes=restored_output.nbytes,batch_blocks=batch,
                **plan.stats())
    return result,restored_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('corpora',type=Path)
    parser.add_argument('--repeats',type=int,default=7);parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--batch-blocks',type=int,default=1024)
    parser.add_argument('--output',type=Path,default=Path('results/lossless/gpu_compute.json'))
    args=parser.parse_args()
    if min(args.repeats,args.width,args.batch_blocks)<1:parser.error('All counts must be positive')
    try:
        import torch
    except ImportError:
        result={'status':'blocked','reason':'PyTorch is not installed; no CUDA benchmark executed.','gpu_measured':False,'public_readiness':'pending GPU evidence'}
        save(args.output,result);print(json.dumps(result));return 2
    if not torch.cuda.is_available():
        result={'status':'blocked','reason':'CUDA GPU is not available; no GPU benchmark executed.','torch':torch.__version__,'gpu_measured':False,'public_readiness':'pending GPU evidence'}
        save(args.output,result);print(json.dumps(result));return 2
    manifest=json.loads(Path('results/lossless/corpus_sources.json').read_text());original={}
    for entry in manifest['files']:
        raw=(args.corpora/entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Source hash mismatch: '+entry['path'])
        original[entry['path']]=raw
    inserted={};replaced={}
    for name,raw in original.items():
        at=len(raw)//3;edited=raw[:at]+b'\nlocal inserted record\n'+raw[at:];inserted[name]=edited
        at=2*len(edited)//3;replaced[name]=edited[:at]+b'updated record'+edited[at+17:]
    workloads={'single_version':{'original':original},
               'edited_versions':{'original':original,'insert':inserted,'replace':replaced}}
    # Warm the device and FFT plans using both paths before collecting timings.
    for versions in workloads.values():
        for reuse in (False,True):measure(torch,versions,reuse,args.width,args.batch_blocks)
    results={};bitwise=True
    for workload,versions in workloads.items():
        samples={'dense':[],'reuse':[]}
        for repeat in range(args.repeats):
            outputs={}
            order=(False,True) if repeat%2==0 else (True,False)
            for reuse in order:
                name='reuse' if reuse else 'dense'
                row,outputs[name]=measure(torch,versions,reuse,args.width,args.batch_blocks)
                samples[name].append(row)
            if not np.array_equal(outputs['dense'],outputs['reuse']):
                bitwise=False
                results[workload]={'status':'failed','reason':'Full GPU outputs differ; no exact-reuse claim allowed.'}
                break
        else:
            medians={name:{key:statistics.median(row[key] for row in rows) for key in rows[0]} for name,rows in samples.items()}
            saving=1-medians['reuse']['total_wall_s']/medians['dense']['total_wall_s']
            results[workload]={'status':'measured','median':medians,'runs':samples,
                              'full_outputs_bitwise_equal':True,'full_inputs_byte_exact':True,
                              'total_wall_reduction_fraction':saving}
    passed=bitwise and all(
        row.get('total_wall_reduction_fraction',-1)>0 and
        row['median']['reuse']['fft_device_ms'] < row['median']['dense']['fft_device_ms']
        for row in results.values())
    result={'status':'measured' if bitwise else 'failed','gpu_measured':True,
            'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'python':platform.python_version(),'source_files':len(original),
            'source_bytes':sum(map(len,original.values())),'width':args.width,'repeats':args.repeats,
            'operation':'float32 rFFT of every zero-padded block; same boundaries and batch cap in both paths',
            'timed_scope':'CPU boundaries, hashing, lookup, packing, H2D, complete FFT, GPU output gather, full D2H, complete original-byte reconstruction and checksums; input disk reads and final independent assertions excluded',
            'limitations':'block-local FFT only; synthetic edits of real corpora; no global-context neural inference claim; no electricity measurement',
            'workloads':results,
            'public_readiness':'eligible for review of scoped GPU claim' if passed else 'hold: exact full-path GPU improvement not demonstrated'}
    save(args.output,result);print(json.dumps(result,indent=2));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
