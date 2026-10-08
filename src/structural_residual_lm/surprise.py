"""Three bounded components; weights are experimental design choices."""
import numpy as np

def surprise(confidence,entropy,mismatch,confidence_enabled=True):
    values=np.asarray([confidence,entropy,mismatch],dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError("Components must be finite and in [0,1]")
    parts=[entropy,mismatch]
    if confidence_enabled: parts.append(1-confidence)
    return float(np.mean(parts))
