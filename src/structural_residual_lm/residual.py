import numpy as np

def residual(observed,predicted):
    a=np.asarray(observed,dtype=float);b=np.asarray(predicted,dtype=float)
    if a.shape!=b.shape: raise ValueError("Shapes must agree")
    return a-b
