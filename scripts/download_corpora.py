"""Fetch exact pinned corpus blobs; verify SHA-256 before saving."""
import argparse,base64,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'results/lossless/corpus_sources.json').read_text())
root=Path(a.directory)
for item in manifest['files']:
    path=root/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:continue
    url='https://api.github.com/repos/'+manifest['repository']+'/git/blobs/'+item['git_blob']
    req=urllib.request.Request(url,headers={'User-Agent':'shared-data-corpus-test'})
    with urllib.request.urlopen(req,timeout=60) as response:blob=json.load(response)
    raw=base64.b64decode(blob['content'])
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    path.write_bytes(raw);print(item['path'],len(raw),flush=True)
