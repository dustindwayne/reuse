import numpy as np

def compute_levels(scores,thresholds=(.25,.50)):
    if len(thresholds)!=2 or not 0<=thresholds[0]<thresholds[1]<=1: raise ValueError("Need two increasing thresholds in [0,1]")
    a=np.asarray(scores,dtype=float)
    if not np.isfinite(a).all() or ((a<0)|(a>1)).any(): raise ValueError("Invalid surprise score")
    return (a>=thresholds[0]).astype(int)+(a>=thresholds[1]).astype(int)
