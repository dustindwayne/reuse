"""Small fixed-context neural LM with real conditional block execution.

Manual NumPy reverse-mode derivatives; no pretrained models or external services.
"""
import numpy as np

class NeuralLM:
    def __init__(self, vocabulary, context=4, embedding=12, hidden=48, features=0, seed=7):
        rng=np.random.default_rng(seed); self.context=context;self.embedding=embedding;self.hidden=hidden
        def weight(a,b): return rng.normal(0,1/np.sqrt(a),(a,b))
        # Shared tensors initialized before feature projection: matched shared initialization.
        self.p={"E":rng.normal(0,.1,(vocabulary,embedding)), "W":weight(context*embedding,hidden), "b":np.zeros(hidden),
                "W0":weight(hidden,hidden), "b0":np.zeros(hidden),
                "W1":weight(hidden,hidden), "b1":np.zeros(hidden),
                "O":weight(hidden,vocabulary), "c":np.zeros(vocabulary)}
        if features: self.p["F"]=weight(features,hidden)
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()};self.step=0

    @property
    def parameter_count(self): return sum(v.size for v in self.p.values())

    def forward(self, x, features=None, levels=None):
        p=self.p;z=p["E"][x].reshape(len(x),-1)
        a=z@p["W"]+p["b"]
        if features is not None: a+=features@p["F"]
        h=np.tanh(a);first=h.copy();cache=[]
        levels=np.full(len(x),2) if levels is None else np.asarray(levels)
        for j in range(2):
            ids=np.flatnonzero(levels>j);before=h[ids].copy()
            # Only selected rows enter matmul; masking full dense computation is avoided.
            delta=np.tanh(before@p[f"W{j}"]+p[f"b{j}"])
            h[ids]+=delta;cache.append((ids,before,delta))
        logits=h@p["O"]+p["c"];logits-=logits.max(axis=1,keepdims=True)
        probs=np.exp(logits);probs/=probs.sum(axis=1,keepdims=True)
        return probs,(x,z,first,h,cache,features)

    def gradients(self,x,y,features=None,levels=None):
        probs,c=self.forward(x,features,levels);idx=np.arange(len(y));loss=float(-np.log(np.maximum(probs[idx,y],1e-300)).mean())
        d=probs.copy();d[idx,y]-=1;d/=len(y)
        x,z,first,h,cache,features=c; p=self.p;g={k:np.zeros_like(v) for k,v in p.items()}
        g["O"]=h.T@d;g["c"]=d.sum(0);dh=d@p["O"].T
        for j in reversed(range(2)):
            ids,before,delta=cache[j]; dd=dh[ids]*(1-delta**2)
            g[f"W{j}"]=before.T@dd;g[f"b{j}"]=dd.sum(0);dh[ids]+=dd@p[f"W{j}"].T
        da=dh*(1-first**2);g["W"]=z.T@da;g["b"]=da.sum(0)
        if features is not None: g["F"]=features.T@da
        de=(da@p["W"].T).reshape(len(x),self.context,self.embedding)
        np.add.at(g["E"],x.reshape(-1),de.reshape(-1,self.embedding))
        return loss,g

    def train_batch(self,x,y,features=None,levels=None,lr=.01):
        loss,g=self.gradients(x,y,features,levels);self.step+=1
        norm=np.sqrt(sum((a*a).sum() for a in g.values()));scale=min(1,5/max(norm,1e-12))
        for k in self.p:
            a=g[k]*scale;self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
            self.p[k]-=lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
        return loss
