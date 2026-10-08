import os,platform,resource,subprocess,sys
import numpy as np

def evaluate(model,x,y,features=None,levels=None,batch=256):
    losses=[];correct=0
    for i in range(0,len(y),batch):
        p,_=model.forward(x[i:i+batch],None if features is None else features[i:i+batch],None if levels is None else levels[i:i+batch])
        target=y[i:i+batch];losses.extend(-np.log(np.maximum(p[np.arange(len(target)),target],1e-300)));correct+=int((p.argmax(1)==target).sum())
    return dict(loss=float(np.mean(losses)),perplexity=float(np.exp(np.mean(losses))),accuracy=correct/len(y))

def environment():
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: commit=None
    dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True)) if commit else None
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(git_commit=commit,git_dirty=dirty,python=sys.version.split()[0],numpy=np.__version__,platform=platform.system(),
                machine=platform.machine(),cpu_threads=os.cpu_count(),device="CPU",peak_ram_mib=peak/(1024 if sys.platform!="darwin" else 1024**2),
                peak_vram_mib=None,vram_note="CPU-only run; GPU allocation not applicable",blas_threads={k:os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]})
