"""Bounded native boundary search and persistent encode/decode reuse."""
import hashlib,json,re,struct,zlib
from .folding import fold,unfold
BOUNDARY=re.compile(rb'\n\n|[\x00\x20\x80\xff][\x00\x20\x80\xff]')


def spans(raw,content=True,minimum=8192,maximum=65536):
    start=0
    while start<len(raw):
        stop=min(len(raw),start+maximum)
        if content and stop-start>minimum:
            match=BOUNDARY.search(raw,start+minimum,stop)
            if match:stop=match.end()
        yield start,stop
        start=stop


def encode_chunk(raw,adaptive=True,level=6):
    width=0;trials=0
    # Bounded sample search; no full-size alternative trials.
    if adaptive and len(raw)>=16384:
        sample=raw[:2048];best=len(zlib.compress(sample,level));trials=1
        for w in (2,4,8):
            cost=1+len(zlib.compress(fold(sample,w),level));trials+=1
            if cost<best*.98:best=cost;width=w
    transformed=fold(raw,width) if width else raw
    compressed=zlib.compress(transformed,level)
    payload=bytes([width])+compressed
    if len(payload)>=len(raw)+1:return b'\0'+raw,trials
    return bytes([1])+payload,trials


def decode_chunk(payload):
    if not payload:raise ValueError('Missing chunk')
    if payload[0]==0:return payload[1:]
    if payload[0]!=1 or len(payload)<2:raise ValueError('Invalid chunk')
    try:raw=zlib.decompress(payload[2:])
    except zlib.error as e:raise ValueError('Invalid compressed chunk') from e
    return unfold(raw,payload[1]) if payload[1] else raw


class Store:
    def __init__(self,content=True,adaptive=False,level=6,whole_file=False):
        self.content=content;self.adaptive=adaptive;self.level=level;self.whole_file=whole_file
        self.chunks=[];self.originals=[];self.index={};self.versions=[]
        self.stats={'input_bytes':0,'encoded_bytes':0,'reused_bytes':0,'sample_trials':0,'full_encodes':0}
    def add(self,name,files):
        entries=[]
        for path,raw in files.items():
            refs=[];self.stats['input_bytes']+=len(raw)
            for start,stop in ([(0,len(raw))] if self.whole_file and raw else spans(raw,self.content)):
                part=raw[start:stop];digest=hashlib.sha256(part).digest()
                if digest in self.index:
                    node=self.index[digest]
                    if self.originals[node]!=part:raise ValueError('Hash collision')
                    self.stats['reused_bytes']+=len(part)
                else:
                    payload,trials=encode_chunk(part,self.adaptive,self.level)
                    node=len(self.chunks);self.index[digest]=node;self.chunks.append((digest,len(part),payload));self.originals.append(part)
                    self.stats['encoded_bytes']+=len(part);self.stats['sample_trials']+=trials;self.stats['full_encodes']+=1
                refs.append(node)
            entries.append({'name':path,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'refs':refs})
        self.versions.append({'name':name,'files':entries})
    def dumps(self):
        manifest=json.dumps(self.versions,ensure_ascii=False,separators=(',',':')).encode()
        out=bytearray(b'FST1'+struct.pack('<II',len(manifest),len(self.chunks))+manifest)
        for digest,size,payload in self.chunks:out+=digest+struct.pack('<II',size,len(payload))+payload
        return bytes(out)


def loads(blob):
    if len(blob)<12 or blob[:4]!=b'FST1':raise ValueError('Invalid store')
    m,n=struct.unpack_from('<II',blob,4);cursor=12+m
    if cursor>len(blob):raise ValueError('Truncated manifest')
    versions=json.loads(blob[12:cursor]);chunks=[]
    for _ in range(n):
        if cursor+40>len(blob):raise ValueError('Truncated chunk header')
        digest=blob[cursor:cursor+32];size,length=struct.unpack_from('<II',blob,cursor+32);cursor+=40
        if cursor+length>len(blob):raise ValueError('Truncated payload')
        raw=decode_chunk(blob[cursor:cursor+length]);cursor+=length
        if len(raw)!=size or hashlib.sha256(raw).digest()!=digest:raise ValueError('Chunk integrity failed')
        chunks.append(raw)
    if cursor!=len(blob):raise ValueError('Trailing bytes')
    out={}
    for version in versions:
        if version['name'] in out:raise ValueError('Duplicate version')
        files={}
        for entry in version['files']:
            if entry['name'] in files or any(type(i)!=int or i<0 or i>=n for i in entry['refs']):raise ValueError('Invalid references')
            raw=b''.join(chunks[i] for i in entry['refs'])
            if len(raw)!=entry['size'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('File integrity failed')
            files[entry['name']]=raw
        out[version['name']]=files
    return out
