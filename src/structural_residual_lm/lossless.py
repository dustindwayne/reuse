"""Lossless byte grammar: unique terminals, shared pairs, ordered root references."""
from collections import Counter
from dataclasses import dataclass


def varint(n):
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


@dataclass
class Grammar:
    terminals: bytes
    pairs: list
    roots: list

    @classmethod
    def build(cls, data, max_merges=128):
        terminals = bytes(sorted(set(data)))
        ids = {b: i for i, b in enumerate(terminals)}
        g = cls(terminals, [], [ids[b] for b in data])
        for _ in range(max_merges):
            counts = Counter(zip(g.roots, g.roots[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda x: (-x[1], x[0]))
            if count < 3:
                break
            roots = []
            i = 0
            node = len(terminals) + len(g.pairs)
            while i < len(g.roots):
                if i + 1 < len(g.roots) and (g.roots[i], g.roots[i+1]) == pair:
                    roots.append(node)
                    i += 2
                else:
                    roots.append(g.roots[i])
                    i += 1
            candidate = cls(terminals, g.pairs + [pair], roots)
            # Count actual dictionary and reference bytes, including ID-width changes.
            if len(candidate.dumps()) >= len(g.dumps()):
                break
            g = candidate
        return g

    def dumps(self):
        out = bytearray(b'LSG1')
        out += varint(len(self.terminals)) + self.terminals
        out += varint(len(self.pairs))
        for a, b in self.pairs:
            out += varint(a) + varint(b)
        out += varint(len(self.roots))
        for node in self.roots:
            out += varint(node)
        return bytes(out)

    @classmethod
    def loads(cls, data):
        if data[:4] != b'LSG1':
            raise ValueError('Invalid grammar header')
        i = 4
        def read():
            nonlocal i
            n = shift = 0
            while i < len(data) and shift <= 63:
                b = data[i]; i += 1
                n |= (b & 127) << shift
                if b < 128:
                    return n
                shift += 7
            raise ValueError('Truncated or oversized integer')
        n = read()
        terminals = data[i:i+n]; i += n
        if len(terminals) != n or len(set(terminals)) != n:
            raise ValueError('Invalid terminals')
        pairs = []
        for _ in range(read()):
            a, b = read(), read()
            if max(a, b) >= n + len(pairs):
                raise ValueError('Non-topological node reference')
            pairs.append((a, b))
        roots = [read() for _ in range(read())]
        if any(r >= n + len(pairs) for r in roots) or i != len(data):
            raise ValueError('Invalid roots or trailing bytes')
        return cls(terminals, pairs, roots)

    def decode(self):
        out = bytearray()
        stack = list(reversed(self.roots))
        while stack:
            node = stack.pop()
            if node < len(self.terminals):
                out.append(self.terminals[node])
            else:
                a, b = self.pairs[node-len(self.terminals)]
                stack.extend((b, a))
        return bytes(out)

    def node_lengths(self):
        """Analyze each definition once, reusing child results."""
        lengths = [1] * len(self.terminals)
        for a, b in self.pairs:
            lengths.append(lengths[a] + lengths[b])
        return lengths
