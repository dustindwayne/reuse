"""Pattern-first archive: prior exact ranges, literal pool, then framework codecs."""
from collections import defaultdict,deque
import hashlib,json,struct,zlib
from .container import select,restore
from .lossless import varint


def factor(raw,seed=16,min_match=32,max_match=65536):
    if seed<1 or min_match<seed or max_match<min_match:raise ValueError('Invalid matching parameters')
    index={};commands=[];literals=bytearray();pending=bytearray();pos=0
    def add(start,end):
        # Sparse prior anchors limit index cost; queries scan every current position.
        for i in range(((start+seed-1)//seed)*seed,end,seed):
            if i+seed>len(raw):break
            key=raw[i:i+seed]
            if key not in index:index[key]=deque(maxlen=8)
            index[key].append(i)
    def flush():
        if pending:
            commands.append((0,len(pending)));literals.extend(pending);pending.clear()
    while pos<len(raw):
        best_source=best_length=0
        if pos+seed<=len(raw):
            for source in index.get(raw[pos:pos+seed],()):
                limit=min(max_match,len(raw)-pos,pos-source)
                if limit<min_match:continue
                length=seed
                while length+1024<=limit and raw[source+length:source+length+1024]==raw[pos+length:pos+length+1024]:length+=1024
                while length<limit and raw[source+length]==raw[pos+length]:length+=1
                if length>best_length:best_source,best_length=source,length
        if best_length>=min_match:
            flush();commands.append((1,best_source,best_length));add(pos,pos+best_length);pos+=best_length
        else:
            pending.append(raw[pos]);add(pos,pos+1);pos+=1
    flush()
    return bytes(literals),commands


def pack(files,matching=True,selector=select):
    raw=b''.join(files.values());entries=[{'name':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()]
    literals,commands=factor(raw) if matching else (raw,[(0,len(raw))] if raw else [])
    instructions=bytearray()
    for command in commands:
        instructions.append(command[0])
        for n in command[1:]:instructions+=varint(n)
    mode,payload=selector(literals)
    manifest=json.dumps(entries,ensure_ascii=False,separators=(',',':')).encode()
    body=struct.pack('<IIIB',len(manifest),len(instructions),len(payload),mode)+manifest+instructions+payload
    compressed=zlib.compress(body)
    return b'PTF1'+bytes([int(len(compressed)<len(body))])+(compressed if len(compressed)<len(body) else body)


def unpack(data,with_stats=False):
    if len(data)<5 or data[:4]!=b'PTF1' or data[4]>1:raise ValueError('Invalid pattern archive')
    try:body=zlib.decompress(data[5:]) if data[4] else data[5:]
    except zlib.error as e:raise ValueError('Corrupt pattern archive') from e
    if len(body)<13:raise ValueError('Truncated header')
    m,c,p,mode=struct.unpack_from('<IIIB',body)
    if 13+m+c+p!=len(body):raise ValueError('Invalid section sizes')
    entries=json.loads(body[13:13+m]);instructions=body[13+m:13+m+c];pool=restore(mode,body[13+m+c:])
    out=bytearray();cursor=offset=refs=shared=0
    def read():
        nonlocal cursor
        n=shift=0
        while cursor<len(instructions) and shift<=63:
            b=instructions[cursor];cursor+=1;n|=(b&127)<<shift
            if b<128:return n
            shift+=7
        raise ValueError('Invalid integer')
    total=sum(e['size'] for e in entries)
    while cursor<len(instructions):
        tag=instructions[cursor];cursor+=1
        if tag==0:
            length=read()
            if offset+length>len(pool):raise ValueError('Invalid literal range')
            out+=pool[offset:offset+length];offset+=length
        elif tag==1:
            source,length=read(),read()
            if source+length>len(out):raise ValueError('Invalid forward/overlapping reference')
            out+=out[source:source+length];refs+=1;shared+=length
        else:raise ValueError('Invalid instruction')
        if len(out)>total:raise ValueError('Output size exceeded')
    if offset!=len(pool) or len(out)!=total:raise ValueError('Output length mismatch')
    files={};offset=0
    for e in entries:
        if e['name'] in files:raise ValueError('Duplicate name')
        raw=bytes(out[offset:offset+e['size']]);offset+=e['size']
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('Integrity failed')
        files[e['name']]=raw
    stats=dict(literal_bytes=len(pool),referenced_bytes=shared,references=refs,instruction_bytes=c,
               literal_codec=mode,outer_zlib=bool(data[4]),uncompressed_container_bytes=len(body)+5)
    return (files,stats) if with_stats else files
