import argparse,base64,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory);root.mkdir(parents=True,exist_ok=True)
m=json.loads((Path(__file__).resolve().parents[1]/'results/lossless/fold_model_sources.json').read_text())
for e in m['files']:
 path=root/e['local_name']
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']:continue
 req=urllib.request.Request('https://api.github.com/repos/'+m['repository']+'/git/blobs/'+e['git_blob'],headers={'User-Agent':'shared-data-corpus-test'})
 with urllib.request.urlopen(req,timeout=60) as response:data=json.load(response)
 raw=base64.b64decode(data['content']);assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];path.write_bytes(raw)
