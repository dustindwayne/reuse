"""Lossless dynamic byte windows with optional frequent-value extraction."""
from collections import Counter
import struct
from .grid import pack_bits, unpack_bits


def average(raw):
    base=sum(raw)//len(raw) if raw else 0
    codes=[2*(v-base) if v>=base else 2*(base-v)-1 for v in raw]
    bits=max(codes,default=0).bit_length()
    return bytes((base,bits))+pack_bits(codes,bits)


def plain(raw):
    return min((b'\x00'+raw,b'\x01'+average(raw)),key=len)


def block(raw, extract=True):
    best=plain(raw)
    if extract:
        counts=Counter(raw); total=sum(raw); mask_size=(len(raw)+7)//8
        winner=None; best_size=len(best)
        # Score candidates from shared counts before constructing any masks.
        for value,count in counts.items():
            if count<2: continue
            n=len(raw)-count
            base=(total-value*count)//n if n else 0
            other=[v for v in counts if v!=value]
            bits=max((2*(v-base) if v>=base else 2*(base-v)-1 for v in other),default=0).bit_length()
            size=2+mask_size+min(1+n,3+(n*bits+7)//8)
            if size<best_size: winner=value; best_size=size
        if winner is not None:
            mask=pack_bits([int(v==winner) for v in raw],1)
            remaining=bytes(v for v in raw if v!=winner)
            best=b'\x02'+bytes((winner,))+mask+plain(remaining)
    return best


def encode(raw, dynamic=True, extract=True, max_window=1024):
    if not 1<=max_window<=65534: raise ValueError('Invalid maximum window')
    def visit(part):
        whole=block(part,extract)
        encoded=struct.pack('<HH',len(part),len(whole))+whole
        if dynamic and len(part)>16:
            middle=len(part)//2
            divided=visit(part[:middle])+visit(part[middle:])
            if len(divided)<len(encoded): return divided
        return encoded
    chunks=[visit(raw[i:i+max_window]) for i in range(0,len(raw),max_window)]
    candidate=b'ADW1\x01'+struct.pack('<Q',len(raw))+b''.join(chunks)
    unchanged=b'ADW1\x00'+raw
    return candidate if len(candidate)<len(unchanged) else unchanged


def restore(payload,n):
    if not payload: raise ValueError('Empty block')
    mode=payload[0]
    if mode==0:
        if len(payload)!=n+1: raise ValueError('Raw length mismatch')
        return payload[1:]
    if mode==1:
        if len(payload)<3: raise ValueError('Missing mean')
        base,bits=payload[1:3]
        if bits>9 or len(payload)!=3+(n*bits+7)//8: raise ValueError('Offset length mismatch')
        values=[base+(c//2 if c%2==0 else -(c//2)-1) for c in unpack_bits(payload[3:],bits,n)]
        if any(v<0 or v>255 for v in values): raise ValueError('Invalid byte')
        return bytes(values)
    if mode==2:
        size=(n+7)//8
        if len(payload)<2+size+1: raise ValueError('Missing extraction mask')
        value=payload[1]; mask=unpack_bits(payload[2:2+size],1,n)
        rest=iter(restore(payload[2+size:],n-sum(mask)))
        return bytes(value if bit else next(rest) for bit in mask)
    raise ValueError('Unknown mode')


def decode(data):
    if len(data)<5 or data[:4]!=b'ADW1': raise ValueError('Invalid header')
    if data[4]==0: return data[5:]
    if data[4]!=1 or len(data)<13: raise ValueError('Invalid mode')
    total=struct.unpack_from('<Q',data,5)[0]; cursor=13; out=bytearray()
    while cursor<len(data):
        if cursor+4>len(data): raise ValueError('Missing window header')
        n,size=struct.unpack_from('<HH',data,cursor); cursor+=4
        if n==0 or cursor+size>len(data) or len(out)+n>total: raise ValueError('Invalid window')
        out+=restore(data[cursor:cursor+size],n); cursor+=size
    if len(out)!=total: raise ValueError('Output length mismatch')
    return bytes(out)
