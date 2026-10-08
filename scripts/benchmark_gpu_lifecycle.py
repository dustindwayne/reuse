"""Complete cold-plus-edit CUDA lifecycles and paired sequence aggregation."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from benchmark_chunk_gpu import sources
from benchmark_gpu import save
from structural_residual_lm.gpu_reuse import BlockPlan, ChunkGPUCache, execute_torch


def totals_summary(rows):
    return dict(median_dense_s=statistics.median(r['dense_s'] for r in rows),
        median_cache_s=statistics.median(r['cache_s'] for r in rows),
        median_paired_saving_s=statistics.median(r['dense_s']-r['cache_s'] for r in rows),
        cache_wins=sum(r['cache_s']<r['dense_s'] for r in rows),
        ties=sum(r['cache_s']==r['dense_s'] for r in rows),pairs=len(rows),runs=rows)


def existing_totals():
    old=json.loads(Path('results/lossless/gpu_chunk_cache.json').read_text())
    phases=[old['cold']['runs'],old['updates']['insert']['runs'],old['updates']['replace']['runs']]
    if len({len(p) for p in phases})!=1:raise ValueError('Sequence counts differ')
    rows=[dict(sequence=i+1,dense_s=sum(p[i]['dense']['total_wall_s'] for p in phases),
               cache_s=sum(p[i]['cache']['total_wall_s'] for p in phases)) for i in range(len(phases[0]))]
    result=totals_summary(rows)
    result['source']='gpu_chunk_cache.json: matching sequence indices; phase times summed per sequence before statistics'
    save(Path('results/lossless/gpu_existing_lifecycle.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


def constructed_versions(original):
    """Eight cumulative synthetic edits, each modifying all 25 complete files."""
    versions=[original]; descriptions=[]
    for step in range(1,9):
        current={}
        for path,raw in versions[-1].items():
            if step%2:
                at=len(raw)//3
                payload=f'\nconstructed lifecycle insertion {step}\n'.encode('ascii')
                edited=raw[:at]+payload+raw[at:]
            else:
                at=2*len(raw)//3
                payload=f'edit-{step:02d}-record'.encode('ascii')
                edited=raw[:at]+payload+raw[at+17:]
            current[path]=edited
        versions.append(current)
        descriptions.append(dict(step=step,kind='insertion' if step%2 else 'replacement',
                                 files_modified=len(current),bytes=sum(map(len,current.values()))))
    return versions,descriptions


def lifecycle(versions,cached,width,batch):
    kernel=lambda rows:torch.fft.rfft(rows.to(torch.float32),dim=1)
    torch.cuda.synchronize()
    start=time.perf_counter()
    cache=ChunkGPUCache('float32-rfft-v1',width,batch_blocks=batch) if cached else None
    outputs=[]; phases=[]
    for step,files in enumerate(versions):
        phase_start=time.perf_counter()
        if cached:
            output,restored,stats=cache.process(files,kernel,'float32-rfft-v1')
        else:
            planning_start=time.perf_counter()
            plan=BlockPlan(width=width,reuse=False);plan.add('current',files)
            planning=time.perf_counter()-planning_start
            output=execute_torch(plan,kernel,batch_blocks=batch).cpu().numpy()
            restored=plan.reconstruct()['current']
            stats=dict(planning_lookup_s=planning,new_blocks=len(plan.blocks),output_blocks=len(plan.refs))
        # The methods themselves reconstruct and verify SHA256 before returning.
        # This additional byte-exact assertion is also inside lifecycle timing.
        if restored!=files:raise ValueError('Full input reconstruction differs')
        torch.cuda.synchronize()
        phases.append(dict(step=step,total_wall_s=time.perf_counter()-phase_start,**stats))
        outputs.append(output)
    # Include releasing GPU cache, index, input views and dense plans.
    del cache
    if not cached:del plan
    del restored
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return dict(total_wall_s=elapsed,phases=phases),outputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('corpora',type=Path,nargs='?',default=Path('data/corpora'))
    parser.add_argument('--existing-only',action='store_true')
    parser.add_argument('--repeats',type=int,default=21)
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--batch-blocks',type=int,default=1024)
    args=parser.parse_args()
    if args.existing_only:existing_totals();return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    versions,edits=constructed_versions(sources(args.corpora)['original'])
    results={}
    # Warmup first; alternate whole lifecycle order, without carrying cache
    # between pairs. Independent full-output equality is outside both timers.
    for count in (1,2,4,8):
        rows=[]
        for repeat in range(args.repeats+1):
            outputs={};measured={}
            gc.collect()
            for cached in ((False,True) if repeat%2==0 else (True,False)):
                method='cache' if cached else 'dense'
                measured[method],outputs[method]=lifecycle(versions[:count+1],cached,args.width,args.batch_blocks)
            for a,b in zip(outputs['dense'],outputs['cache']):
                if not np.array_equal(a,b):raise ValueError('Full GPU outputs differ')
            if repeat:rows.append(dict(sequence=repeat,dense_s=measured['dense']['total_wall_s'],
                cache_s=measured['cache']['total_wall_s'],dense_phases=measured['dense']['phases'],
                cache_phases=measured['cache']['phases']))
        result=totals_summary(rows)
        result['full_outputs_bitwise_equal']=True
        result['full_inputs_byte_exact']=True
        results[str(count)]=result
        print('edits',count,json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)
    save(Path('results/lossless/gpu_lifecycle.json'),dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,repeats=args.repeats,width=args.width,
        batch_blocks=args.batch_blocks,operation='Unchanged float32 rFFT per zero-padded block',
        edits_label='Constructed cumulative edits of all 25 pinned corpus files, not observed user editing history',
        constructed_edits=edits,results=results,
        timing_scope='Complete lifecycle: cold cache construction, every planning/lookup, packing, H2D, unchanged FFT, GPU output maintenance and gathering, full D2H outputs, full byte reconstruction and SHA256 verification, cache teardown. Source reads, synthetic edit generation, independent full output comparison and pre-run GC excluded.'))


if __name__=='__main__':main()
