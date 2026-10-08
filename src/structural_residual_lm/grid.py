"""Exact byte-grid blocks: integer mean plus bit-packed signed offsets."""
import struct

HEADER = struct.Struct('<4sQIII')


def blocks(n, width, bw, bh):
    rows = (n + width - 1) // width
    for y in range(0, rows, bh):
        for x in range(0, width, bw):
            ids = [r*width+c for r in range(y, min(y+bh, rows))
                   for c in range(x, min(x+bw, width)) if r*width+c < n]
            if ids:
                yield ids


def pack_bits(values, bits):
    out = bytearray(); acc = count = 0
    for value in values:
        acc |= value << count; count += bits
        while count >= 8:
            out.append(acc & 255); acc >>= 8; count -= 8
    if count:
        out.append(acc)
    return bytes(out)


def unpack_bits(data, bits, n):
    acc = int.from_bytes(data, 'little'); mask = (1 << bits)-1
    return [(acc >> (i*bits)) & mask for i in range(n)]


def encode(raw, width=64, bw=8, bh=8, adaptive=True):
    if min(width, bw, bh) <= 0:
        raise ValueError('Dimensions must be positive')
    out = bytearray(HEADER.pack(b'GRD1', len(raw), width, bw, bh))
    for ids in blocks(len(raw), width, bw, bh):
        values = [raw[i] for i in ids]
        base = sum(values)//len(values)
        delta = [v-base for v in values]
        codes = [2*d if d >= 0 else -2*d-1 for d in delta]
        bits = max(codes).bit_length()
        average = bytes((1, base, bits)) + pack_bits(codes, bits)
        unchanged = b'\x00' + bytes(values)
        out += unchanged if adaptive and len(unchanged) <= len(average) else average
    return bytes(out)


def decode(packed):
    if len(packed) < HEADER.size:
        raise ValueError('Truncated header')
    magic, n, width, bw, bh = HEADER.unpack_from(packed)
    if magic != b'GRD1' or min(width, bw, bh) == 0:
        raise ValueError('Invalid header')
    # Every nonempty block takes at least one byte; reject impossible lengths.
    if n > (len(packed)-HEADER.size)*bw*bh:
        raise ValueError('Impossible output size')
    out = bytearray(n); cursor = HEADER.size
    for ids in blocks(n, width, bw, bh):
        if cursor >= len(packed):
            raise ValueError('Truncated block')
        mode = packed[cursor]; cursor += 1
        if mode == 0:
            size = len(ids); values = packed[cursor:cursor+size]
        elif mode == 1:
            if cursor+2 > len(packed):
                raise ValueError('Truncated average')
            base, bits = packed[cursor:cursor+2]; cursor += 2
            if bits > 9:
                raise ValueError('Invalid offset width')
            size = (len(ids)*bits+7)//8
            codes = unpack_bits(packed[cursor:cursor+size], bits, len(ids))
            values = [base + (c//2 if c%2 == 0 else -(c//2)-1) for c in codes]
            if any(v < 0 or v > 255 for v in values):
                raise ValueError('Invalid reconstructed byte')
        else:
            raise ValueError('Invalid mode')
        if cursor+size > len(packed):
            raise ValueError('Truncated payload')
        for i, value in zip(ids, values):
            out[i] = value
        cursor += size
    if cursor != len(packed):
        raise ValueError('Trailing data')
    return bytes(out)
