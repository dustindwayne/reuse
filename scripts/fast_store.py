"""Store complete directory versions with shared, exactly reversible chunks."""
import argparse,sys
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_residual_lm.fast import Store,loads

def safe_path(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('..','.') for x in p.parts) or '\\' in name:raise ValueError('Unsafe stored path')
    return p

def main():
    parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='command',required=True)
    pack=subs.add_parser('pack');pack.add_argument('output',type=Path);pack.add_argument('directories',nargs='+',type=Path);pack.add_argument('--sample-folding',action='store_true')
    unpack=subs.add_parser('unpack');unpack.add_argument('input',type=Path);unpack.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    if args.command=='pack':
        store=Store(adaptive=args.sample_folding);names=set()
        for directory in args.directories:
            if not directory.is_dir():parser.error(f'Not a directory: {directory}')
            name=directory.resolve().name;safe_path(name)
            if name in names:parser.error('Version directory names must be distinct')
            names.add(name);files={}
            for path in sorted(directory.rglob('*')):
                if path.is_symlink():parser.error(f'Symlinks are unsupported: {path}')
                if path.is_file():files[path.relative_to(directory).as_posix()]=path.read_bytes()
            store.add(name,files)
        blob=store.dumps()
        with args.output.open('xb') as handle:handle.write(blob)
        print(f'{len(blob)} stored bytes; {store.stats}')
    else:
        versions=loads(args.input.read_bytes())
        # Validate every destination before creating any output.
        for version,files in versions.items():
            safe_path(version)
            for name in files:safe_path(name)
        args.output.mkdir()
        for version,files in versions.items():
            root=args.output/version;root.mkdir()
            for name,raw in files.items():
                destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
if __name__=='__main__':main()
