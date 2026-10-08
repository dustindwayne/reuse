"""Unified lossless archive with shared chunks and adaptive codec selection."""
import hashlib,json,struct,zlib
from .adaptive import encode as adaptive_encode, decode as adaptive_decode
from .grid import encode as grid_encode, decode as grid_decode
from .lossless import Grammar


def select(raw):
    candidates=[(0,raw),(1,zlib.compress(raw)),(2,adaptive_encode(raw)),
                (3,Grammar.build(raw,max_merges=32).dumps()),(4,grid_encode(raw))]
    candidates += [(mode+3,zlib.compress(payload)) for mode,payload in candidates if mode in (2,3,4)]
    mode,payload=min(candidates,key=lambda x:len(x[1]))
    return mode,payload


def restore(mode,payload):
    if mode in (5,6,7):
        try: transformed=zlib.decompress(payload)
        except zlib.error as e: raise ValueError('Invalid compressed transform') from e
        return restore(mode-3,transformed)
    if mode==0: return payload
    if mode==1:
        try: return zlib.decompress(payload)
        except zlib.error as e: raise ValueError('Invalid compressed payload') from e
    if mode==2: return adaptive_decode(payload)
    if mode==3: return Grammar.loads(payload).decode()
    if mode==4: return grid_decode(payload)
    raise ValueError('Unknown codec')


def pack(files,chunk_size=4096):
    if chunk_size<=0: raise ValueError('Invalid chunk size')
    chunks=[]; originals=[]; seen={}; entries=[]
    for name,raw in files.items():
        refs=[]
        for i in range(0,len(raw),chunk_size):
            chunk=raw[i:i+chunk_size]
            digest=hashlib.sha256(chunk).digest()
            if digest not in seen:
                mode,payload=select(chunk)
                seen[digest]=len(chunks)
                chunks.append((digest,len(chunk),mode,payload)); originals.append(chunk)
            else:
                # A hash identifies candidates; exact bytes establish equality.
                if originals[seen[digest]]!=chunk: raise ValueError('Digest collision')
            refs.append(seen[digest])
        entries.append(dict(name=name,size=len(raw),sha256=hashlib.sha256(raw).hexdigest(),refs=refs))
    manifest=json.dumps(entries,ensure_ascii=False,separators=(',',':')).encode()
    out=bytearray(b'UDF1'+struct.pack('<II',len(manifest),len(chunks))+manifest)
    for digest,n,mode,payload in chunks:
        out+=digest+struct.pack('<IBI',n,mode,len(payload))+payload
    return bytes(out)


def unpack(data,with_stats=False):
    if len(data)<12 or data[:4]!=b'UDF1': raise ValueError('Invalid container')
    m,n=struct.unpack_from('<II',data,4); cursor=12+m
    if cursor>len(data): raise ValueError('Truncated manifest')
    entries=json.loads(data[12:cursor]); chunks=[]; modes={}; payload_bytes=0
    for _ in range(n):
        if cursor+41>len(data): raise ValueError('Truncated chunk header')
        digest=data[cursor:cursor+32]; size,mode,length=struct.unpack_from('<IBI',data,cursor+32); cursor+=41
        if cursor+length>len(data): raise ValueError('Truncated chunk')
        raw=restore(mode,data[cursor:cursor+length]);cursor+=length
        if len(raw)!=size or hashlib.sha256(raw).digest()!=digest: raise ValueError('Chunk integrity failed')
        chunks.append(raw);modes[str(mode)]=modes.get(str(mode),0)+1;payload_bytes+=length
    if cursor!=len(data): raise ValueError('Trailing bytes')
    files={}
    for entry in entries:
        if entry['name'] in files or any(type(r)!=int or r<0 or r>=len(chunks) for r in entry['refs']):
            raise ValueError('Invalid file references')
        raw=b''.join(chunks[r] for r in entry['refs'])
        if len(raw)!=entry['size'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']: raise ValueError('File integrity failed')
        files[entry['name']]=raw
    stats=dict(unique_chunks=n,total_references=sum(len(e['refs']) for e in entries),codec_counts=modes,
               payload_bytes=payload_bytes,metadata_bytes=len(data)-payload_bytes)
    return (files,stats) if with_stats else files
