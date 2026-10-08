"""Reversible byte-lane folding and recovered 3/1/3 permutation."""

def fold(raw,width):
    if not 1<=width<=255:raise ValueError('Invalid fold width')
    return b''.join(raw[r::width] for r in range(width))


def unfold(raw,width):
    if not 1<=width<=255:raise ValueError('Invalid fold width')
    out=bytearray(len(raw));cursor=0
    for r in range(width):
        n=max(0,(len(raw)+width-1-r)//width)
        out[r::width]=raw[cursor:cursor+n];cursor+=n
    return bytes(out)


def triplet313(raw):
    return b''.join(raw[i:i+3] for i in range(0,len(raw),7))+raw[3::7]+b''.join(raw[i+4:i+7] for i in range(0,len(raw),7))


def untriplet313(raw):
    q,r=divmod(len(raw),7);a=3*q+min(r,3);b=q+int(r>3)
    left,center,right=raw[:a],raw[a:a+b],raw[a+b:]
    out=bytearray();li=ri=ci=0
    for i in range(0,len(raw),7):
        n=min(7,len(raw)-i);nl=min(3,n);out+=left[li:li+nl];li+=nl
        if n>3:out.append(center[ci]);ci+=1
        nr=max(0,n-4);out+=right[ri:ri+nr];ri+=nr
    return bytes(out)


def candidates(raw):
    import zlib
    return [(8,bytes([w])+zlib.compress(fold(raw,w))) for w in (2,4,8,16,32,64,128)]+[(9,zlib.compress(triplet313(raw)))]
