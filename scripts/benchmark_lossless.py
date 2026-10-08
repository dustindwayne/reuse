"""Compare full serialized grammar with raw bytes and standard compression."""
import hashlib,json,sys,time,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.lossless import Grammar

def measure(name, raw):
    t=time.perf_counter(); g=Grammar.build(raw); build=time.perf_counter()-t
    packed=g.dumps(); t=time.perf_counter(); decoded=Grammar.loads(packed).decode(); decode=time.perf_counter()-t
    assert decoded==raw
    lengths=g.node_lengths()
    assert sum(lengths[r] for r in g.roots)==len(raw)
    return dict(name=name,input_bytes=len(raw),grammar_bytes=len(packed),zlib_bytes=len(zlib.compress(raw)),
                unique_byte_definitions=len(g.terminals),pair_definitions=len(g.pairs),root_references=len(g.roots),
                definition_analysis_visits=len(lengths),build_seconds=build,decode_seconds=decode,
                sha256=hashlib.sha256(raw).hexdigest(),exact_roundtrip=True)

if __name__=='__main__':
    rows=[measure('s_950000',b's'*950000),measure('mixed_exact_text','S s\t café\r\n𐍈 s!'.encode()),
          measure('synthetic_train',Path('data/train.txt').read_bytes())]
    Path('results/lossless/benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
