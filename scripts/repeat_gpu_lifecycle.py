"""Run the frozen lifecycle benchmark in fresh processes and preserve evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'results/lossless/gpu_lifecycle_frozen_sessions'


def hashes():
    paths=list((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'scripts'/name for name in ('benchmark_gpu_lifecycle.py',
               'benchmark_chunk_gpu.py','benchmark_gpu.py')]
    paths += [ROOT/'results/lossless/corpus_sources.json']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('session',type=int,choices=(1,2,3))
    parser.add_argument('--archive-dir',type=Path,default=ARCHIVE,
                        help='Use a new directory to preserve the committed benchmark sessions')
    args=parser.parse_args()
    archive=args.archive_dir.resolve()
    archive.mkdir(parents=True,exist_ok=True)
    frozen=archive/'frozen_manifest.json'
    destination=archive/f'session_{args.session}.json'
    if destination.exists():raise RuntimeError('Session already exists; refusing to overwrite')
    if args.session==1:
        if frozen.exists():raise RuntimeError('Freeze manifest already exists')
        frozen.write_text(json.dumps(hashes(),indent=2)+'\n')
        shutil.copy2(ROOT/'results/lossless/gpu_lifecycle.json',archive/'original_lifecycle.json')
    expected=json.loads(frozen.read_text())
    if hashes()!=expected:raise RuntimeError('Frozen source hashes changed')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    command=[sys.executable,str(ROOT/'scripts/benchmark_gpu_lifecycle.py'),'data/corpora',
             '--repeats','21','--width','2048','--batch-blocks','1024']
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (archive/f'session_{args.session}.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(f'Session failed: {run.stderr}')
    if hashes()!=expected:raise RuntimeError('Frozen source hashes changed during session')
    shutil.copy2(ROOT/'results/lossless/gpu_lifecycle.json',destination)
    shutil.copy2(archive/'original_lifecycle.json',ROOT/'results/lossless/gpu_lifecycle.json')
    result=json.loads(destination.read_text())
    for count,row in result['results'].items():
        if len(row['runs'])!=21:raise RuntimeError('Wrong sequence count')
        if not row['full_outputs_bitwise_equal'] or not row['full_inputs_byte_exact']:
            raise RuntimeError('Exactness gate failed')
        savings=[r['dense_s']-r['cache_s'] for r in row['runs']]
        if sum(s>0 for s in savings)!=row['cache_wins']:raise RuntimeError('Win count mismatch')
    metadata=dict(session=args.session,started_utc=started,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  independent_process=True,command=command,source_hashes_unchanged=True)
    (archive/f'session_{args.session}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({count:{'median_dense_ms':1000*row['median_dense_s'],
         'median_cache_ms':1000*row['median_cache_s'],
         'median_paired_saving_ms':1000*row['median_paired_saving_s'],
         'cache_wins':row['cache_wins'],'pairs':row['pairs']}
         for count,row in result['results'].items()},indent=2))


if __name__=='__main__':main()
