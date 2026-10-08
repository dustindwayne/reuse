"""Exact byte-block references for deterministic, block-local GPU operations.

Planning uses CPU bytes. A kernel must depend only on its padded input row and
an immutable configuration; global attention and stateful kernels do not qualify.
"""
import hashlib
import numpy as np
from .fast import spans


class BlockPlan:
    def __init__(self, width=2048, reuse=True):
        if width < 1:
            raise ValueError('Block width must be positive')
        self.width = width
        self.reuse = reuse
        self.blocks = []
        self.index = {}
        self.refs = []
        self.lengths = []
        self.versions = []
        self.input_bytes = 0

    def add(self, name, files):
        if any(version['name'] == name for version in self.versions):
            raise ValueError('Duplicate version')
        entries = []
        for path, raw in files.items():
            first = len(self.refs)
            self.input_bytes += len(raw)
            for start, stop in spans(raw):
                for offset in range(start, stop, self.width):
                    part = raw[offset:min(offset+self.width, stop)]
                    digest = hashlib.sha256(part).digest()
                    node = self.index.get(digest) if self.reuse else None
                    if node is None:
                        node = len(self.blocks)
                        self.blocks.append(part)
                        if self.reuse:
                            self.index[digest] = node
                    elif self.blocks[node] != part:
                        raise ValueError('Hash collision')
                    self.refs.append(node)
                    self.lengths.append(len(part))
            entries.append(dict(name=path, first=first, stop=len(self.refs),
                                size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        self.versions.append(dict(name=name, files=entries))

    def inputs(self, first=0, stop=None):
        blocks = self.blocks[first:stop]
        if not blocks:
            return np.empty((0, self.width), dtype=np.uint8)
        packed = b''.join(part.ljust(self.width, b'\0') for part in blocks)
        return np.frombuffer(packed, dtype=np.uint8).reshape(-1, self.width).copy()

    def reconstruct(self):
        result = {}
        for version in self.versions:
            files = {}
            for entry in version['files']:
                raw = b''.join(self.blocks[self.refs[i]] for i in range(entry['first'], entry['stop']))
                if len(raw) != entry['size'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError('Reconstruction integrity failed')
                files[entry['name']] = raw
            result[version['name']] = files
        return result

    def stats(self):
        distinct_bytes = sum(map(len, self.blocks))
        return dict(input_bytes=self.input_bytes, computed_input_bytes=distinct_bytes,
                    reused_bytes=self.input_bytes-distinct_bytes,
                    computed_blocks=len(self.blocks), output_blocks=len(self.refs))


def execute_numpy(plan, kernel):
    """Reference backend for testing the same lookup/gather contract."""
    unique = kernel(plan.inputs())
    if len(unique) != len(plan.blocks):
        raise ValueError('Kernel must produce one output per input block')
    return unique[np.asarray(plan.refs, dtype=np.int64)]


def execute_torch(plan, kernel, device='cuda', batch_blocks=1024, events=None):
    """Compute unique rows once, then gather every full output row on device.

    Cache scope is one invocation, so changing the kernel cannot reuse stale
    outputs. Each batch uses the same kernel and configuration. Torch is optional.
    """
    import torch
    if batch_blocks < 1:
        raise ValueError('Batch size must be positive')
    if not plan.blocks:
        raise ValueError('GPU benchmark requires nonempty input')
    outputs = []
    with torch.inference_mode():
        for first in range(0, len(plan.blocks), batch_blocks):
            rows = torch.from_numpy(plan.inputs(first, first+batch_blocks)).to(device)
            pair = None
            if events is not None:
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                pair[0].record()
            output = kernel(rows)
            if pair is not None:
                pair[1].record()
                events.append(pair)
            if len(output) != len(rows):
                raise ValueError('Kernel must produce one output per input block')
            outputs.append(output)
        unique = torch.cat(outputs, dim=0)
        refs = torch.tensor(plan.refs, dtype=torch.int64, device=device)
        return unique.index_select(0, refs)
