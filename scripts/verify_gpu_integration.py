"""Check committed evidence and CUDA exactness without extending benchmarks."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def verify_evidence():
    archive=ROOT/'results/lossless/gpu_lifecycle_frozen_sessions'
    manifest=json.loads((archive/'frozen_manifest.json').read_text())
    for path,expected in manifest.items():
        if hashlib.sha256((ROOT/path.replace('\\','/')).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen source differs: '+path)
    pairs=0
    for session in (1,2,3):
        result=json.loads((archive/f'session_{session}.json').read_text())
        if (result['repeats'],result['width'],result['batch_blocks'])!=(21,2048,1024):
            raise ValueError('Recorded configuration differs')
        for row in result['results'].values():
            runs=row['runs'];differences=[r['dense_s']-r['cache_s'] for r in runs]
            if len(runs)!=21 or statistics.median(differences)!=row['median_paired_saving_s']:
                raise ValueError('Recorded paired savings differ')
            if sum(v>0 for v in differences)!=row['cache_wins']:
                raise ValueError('Recorded win count differs')
            for method in ('dense','cache'):
                if statistics.median(r[method+'_s'] for r in runs)!=row['median_'+method+'_s']:
                    raise ValueError('Recorded median differs')
            if not row['full_outputs_bitwise_equal'] or not row['full_inputs_byte_exact']:
                raise ValueError('Recorded exactness gate failed')
            pairs+=len(runs)
    print(f'Freeze source hashes and all {pairs} recorded lifecycle pairs verified.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpora',type=Path)
    args=parser.parse_args();verify_evidence()
    try:
        import torch
    except ImportError:
        print('CUDA exactness smoke skipped: optional Torch unavailable.');return
    if not torch.cuda.is_available():
        print('CUDA exactness smoke skipped: CUDA unavailable.');return
    from benchmark_gpu_lifecycle import constructed_versions
    from benchmark_chunk_gpu import sources
    from structural_residual_lm.gpu_reuse import BlockPlan,ChunkGPUCache,execute_torch
    original=(sources(args.corpora)['original'] if args.corpora else
              {'data':np.random.default_rng(19).integers(0,256,120000,dtype=np.uint8).tobytes(),
               'tail':b'A\0','empty':b''})
    versions,_=constructed_versions(original)
    kernel=lambda rows:torch.fft.rfft(rows.to(torch.float32),dim=1)
    cache=ChunkGPUCache('float32-rfft-v1',width=2048,batch_blocks=1024)
    for files in versions:
        plan=BlockPlan(width=2048,reuse=False);plan.add('current',files)
        expected=execute_torch(plan,kernel,batch_blocks=1024).cpu().numpy()
        actual,restored,_=cache.process(files,kernel,'float32-rfft-v1')
        if not np.array_equal(actual,expected) or restored!=files or plan.reconstruct()['current']!=files:
            raise ValueError('Full CUDA output or reconstructed input differs')
    torch.cuda.synchronize()
    print(f'CUDA exactness passed on {torch.cuda.get_device_name()}: {len(versions)} full versions; no new timing results.')


if __name__=='__main__':main()
