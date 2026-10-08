"""Select complete archive paths, preserving file boundaries when useful."""
import zlib
from .container import pack as chunk_pack,unpack as chunk_unpack
from .patterns import pack as pattern_pack,unpack as pattern_unpack


def filewise(files):
    inner=chunk_pack(files,chunk_size=max(map(len,files.values()),default=1) or 1)
    compressed=zlib.compress(inner)
    return b'FAW1'+bytes([int(len(compressed)<len(inner))])+(compressed if len(compressed)<len(inner) else inner)


def unpack(data):
    if data[:4]==b'PTF1':return pattern_unpack(data)
    if data[:4]==b'UDF1':return chunk_unpack(data)
    if data[:4]==b'FAW1' and len(data)>=5 and data[4] in (0,1):
        try:inner=zlib.decompress(data[5:]) if data[4] else data[5:]
        except zlib.error as e:raise ValueError('Invalid outer stream') from e
        return chunk_unpack(inner)
    raise ValueError('Unknown archive format')


def pack(files):
    candidates=[pattern_pack(files,matching=False),pattern_pack(files),filewise(files)]
    return min(candidates,key=len)
