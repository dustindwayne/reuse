import hashlib,json,re
from pathlib import Path
import numpy as np

TOKEN_PATTERN=re.compile(r"\w+|[^\w\s]",re.UNICODE)
def tokenize(line): return ["<bos>"]+TOKEN_PATTERN.findall(line.lower())+["<eos>"]

def prepare(output="data",seed=19,input_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    if input_path:
        lines=list(dict.fromkeys(l.strip() for l in Path(input_path).read_text().splitlines() if l.strip()))
        origin="user supplied line documents"
    else:
        # A designed smoke corpus, not natural-language benchmark evidence.
        subjects=["the frog","a bird","the child","a painter","the gardener","a robot","the teacher","a fox"]
        verbs=["watches","finds","moves","follows","sees","carries","builds","studies"]
        objects=["the stone","a branch","the mushroom","a bright flower","the small box","a red ball","the green leaf","a wooden frame"]
        endings=["near the river","under the tree","in the garden","before the rain","after the storm","by the old house"]
        lines=[f"{s} {v} {o} {e}." for s in subjects for v in verbs for o in objects for e in endings]
        origin="authored combinatorial grammar smoke corpus (MIT)"
    if len(lines)<20: raise ValueError("Need at least 20 unique line documents")
    order=rng.permutation(len(lines)); cut=int(.8*len(lines))
    for name,indices in [("train",order[:cut]),("validation",order[cut:])]:
        (out/(name+".txt")).write_text("\n".join(lines[i] for i in indices)+"\n")
    meta={"origin":origin,"seed":seed,"split":"80/20 unique documents; boundaries retained","documents":len(lines),
          "hashes":{n:hashlib.sha256((out/(n+".txt")).read_bytes()).hexdigest() for n in ["train","validation"]}}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta

def load(directory="data",context=4):
    root=Path(directory);train=[tokenize(l) for l in (root/"train.txt").read_text().splitlines()];val=[tokenize(l) for l in (root/"validation.txt").read_text().splitlines()]
    words=sorted({t for seq in train for t in seq});vocab=["<unk>"]+words;ids={t:i for i,t in enumerate(vocab)}
    def examples(sequences):
        x=[];y=[];locations=[]
        for doc,seq in enumerate(sequences):
            tokens=[ids.get(t,0) for t in seq]
            for i in range(1,len(seq)):
                history=[ids["<bos>"]]*max(0,context-i)+tokens[max(0,i-context):i]
                x.append(history);y.append(tokens[i]);locations.append((doc,i))
        return np.asarray(x),np.asarray(y),locations
    return dict(train_sequences=train,validation_sequences=val,vocab=vocab,train=examples(train),validation=examples(val),
                hashes={n:hashlib.sha256((root/(n+".txt")).read_bytes()).hexdigest() for n in ["train","validation"]})
