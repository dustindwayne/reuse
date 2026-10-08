"""Pack files into one archive or extract with exact content verification."""
import argparse,sys
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.container import pack,unpack
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='op',required=True)
a=sub.add_parser('pack');a.add_argument('output');a.add_argument('files',nargs='+');a.add_argument('--patterns',action='store_true')
a=sub.add_parser('unpack');a.add_argument('archive');a.add_argument('directory')
a=p.parse_args()
if a.op=='pack':
    files={}
    for filename in a.files:
        path=Path(filename)
        if path.name in files: raise SystemExit('Duplicate basename')
        files[path.name]=path.read_bytes()
    if a.patterns:
        from structural_residual_lm.patterns import pack as pattern_pack
        data=pattern_pack(files)
    else: data=pack(files)
    with Path(a.output).open('xb') as f:f.write(data)
    print(f'{sum(map(len,files.values()))} input bytes -> {len(data)} archive bytes')
else:
    blob=Path(a.archive).read_bytes()
    if blob[:4]==b'PTF1':
        from structural_residual_lm.patterns import unpack as pattern_unpack
        files=pattern_unpack(blob)
    else: files=unpack(blob)
    root=Path(a.directory)
    if root.exists():raise SystemExit('Output directory must not exist')
    for name in files:
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise SystemExit('Unsafe archive path')
    root.mkdir(parents=True)
    for name,data in files.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
