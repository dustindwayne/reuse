"""Profile block planning, then compare exact chunk caching and dense CUDA."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from benchmark_gpu import measure, save


def sources(root):
    manifest=json.loads(Path('results/lossless/corpus_sources.json').read_text())
    original={}
    for entry in manifest['files']:
        raw=(root/entry['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        original[entry['path']]=raw
    inserted={}; replaced={}
    for name,raw in original.items():
        at=len(raw)//3
        edited=raw[:at]+b'\nlocal inserted record\n'+raw[at:]
        inserted[name]=edited
        at=2*len(edited)//3
        replaced[name]=edited[:at]+b'updated record'+edited[at+17:]
    return {'original':original,'insert':inserted,'replace':replaced}


def summarize(rows):
    medians={method:{key:statistics.median(r[method][key] for r in rows)
                     for key in rows[0][method]} for method in ('dense','cache')}
    return dict(median=medians,median_paired_wall_difference_s=statistics.median(
        r['dense']['total_wall_s']-r['cache']['total_wall_s'] for r in rows),
        wall_reduction_fraction=1-medians['cache']['total_wall_s']/medians['dense']['total_wall_s'],
        full_outputs_bitwise_equal=True,full_inputs_byte_exact=True,runs=rows)


def profile(versions,repeats,width,batch):
    rows=[]
    for repeat in range(repeats+1):
        measured={};outputs={}
        for reuse in ((False,True) if repeat%2==0 else (True,False)):
            key='cache' if reuse else 'dense'
            measured[key],outputs[key]=measure(torch,{'current':versions['original']},reuse,width,batch)
        assert np.array_equal(outputs['dense'],outputs['cache'])
        if repeat:rows.append(measured)
    return summarize(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('corpora',type=Path)
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--repeats',type=int,default=21)
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--batch-blocks',type=int,default=1024)
    args=parser.parse_args()
    assert torch.cuda.is_available()
    versions=sources(args.corpora)
    if args.profile_only:
        result=profile(versions,args.repeats,args.width,args.batch_blocks)
        save(Path('results/lossless/gpu_block_profile.json'),result)
        print(json.dumps(result['median'],indent=2))
        print('paired wall difference ms',1000*result['median_paired_wall_difference_s'])
        return
    from structural_residual_lm.gpu_reuse import ChunkGPUCache
    kernel=lambda rows:torch.fft.rfft(rows.to(torch.float32),dim=1)
    phases={name:[] for name in versions}
    for repeat in range(args.repeats+1):
        cache=None
        for name,files in versions.items():
            measurements={};outputs={}
            for cached in ((False,True) if repeat%2==0 else (True,False)):
                if cached:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    if cache is None:
                        cache=ChunkGPUCache('float32-rfft-v1',args.width,batch_blocks=args.batch_blocks)
                    events=[]
                    outputs['cache'],restored,stats=cache.process(files,kernel,'float32-rfft-v1',events=events)
                    torch.cuda.synchronize()
                    measurements['cache']=dict(total_wall_s=time.perf_counter()-start,
                        fft_device_ms=sum(a.elapsed_time(b) for a,b in events),
                        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),**stats)
                    assert restored==files
                else:
                    measurements['dense'],outputs['dense']=measure(torch,{'current':files},False,args.width,args.batch_blocks)
            assert np.array_equal(outputs['dense'],outputs['cache'])
            if repeat:phases[name].append(measurements)
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        repeats=args.repeats,width=args.width,batch_blocks=args.batch_blocks,
        source_files=len(versions['original']),source_bytes=sum(map(len,versions['original'].values())),
        operation='Unchanged float32 block-local rFFT',
        timing_scope='Planning, chunk lookup, packing, H2D, FFT, GPU cache append/gather, complete D2H, reconstruction and SHA256 verification; disk reads and independent comparison excluded',
        cold=summarize(phases['original']),updates={name:summarize(phases[name]) for name in ('insert','replace')})
    save(Path('results/lossless/gpu_chunk_cache.json'),result)
    print(json.dumps({name:summarize(rows)['median'] for name,rows in phases.items()},indent=2))


if __name__=='__main__':main()
