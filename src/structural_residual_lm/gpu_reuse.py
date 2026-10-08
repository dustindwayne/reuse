"""Exact byte-block references for deterministic, block-local GPU operations.

Planning uses CPU bytes. A kernel must depend only on its padded input row and
an immutable configuration; global attention and stateful kernels do not qualify.
"""
import hashlib
import bisect
import time
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
                    part = memoryview(raw)[offset:min(offset+self.width, stop)]
                    digest = hashlib.sha256(part).digest() if self.reuse else None
                    node = self.index.get(digest) if self.reuse else None
                    if node is None:
                        node = len(self.blocks)
                        self.blocks.append(part)
                        if self.reuse:
                            self.index[digest] = node
                    elif bytes(self.blocks[node]) != bytes(part):
                        raise ValueError('Hash collision')
                    self.refs.append(node)
                    self.lengths.append(len(part))
            entries.append(dict(name=path, first=first, stop=len(self.refs),
                                size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        self.versions.append(dict(name=name, files=entries))

    def inputs(self, first=0, stop=None, buffer=None):
        blocks = self.blocks[first:stop]
        if not blocks:
            return np.empty((0, self.width), dtype=np.uint8)
        packed = np.empty((len(blocks), self.width), dtype=np.uint8) if buffer is None else buffer[:len(blocks)]
        packed.fill(0)
        for row, part in zip(packed, blocks):
            row[:len(part)] = np.frombuffer(part, dtype=np.uint8)
        return packed

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


def execute_torch(plan, kernel, device='cuda', batch_blocks=1024, events=None, pinned=False):
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
    staging = (torch.empty((min(batch_blocks,len(plan.blocks)),plan.width),dtype=torch.uint8,
                           pin_memory=True) if pinned else None)
    buffer = staging.numpy() if pinned else np.empty((min(batch_blocks,len(plan.blocks)),plan.width),dtype=np.uint8)
    with torch.inference_mode():
        for first in range(0, len(plan.blocks), batch_blocks):
            if pinned and first:
                torch.cuda.current_stream().synchronize()
            packed = plan.inputs(first, first+batch_blocks,buffer)
            rows = torch.from_numpy(packed).to(device,non_blocking=pinned)
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
        if pinned:
            torch.cuda.current_stream().synchronize()
        unique = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if len(plan.refs) == len(plan.blocks) and all(i == ref for i, ref in enumerate(plan.refs)):
            return unique
        refs = torch.tensor(plan.refs, dtype=torch.int64, device=device)
        return unique.index_select(0, refs)


class BlockGPUCache:
    """Retain verified input views and outputs for one immutable kernel identity.

    Callers must change kernel_id whenever operation parameters change.
    Configuration changes clear both the byte index and computed outputs.
    """
    plan_type = BlockPlan

    def __init__(self, kernel_id, width=2048, device='cuda', batch_blocks=1024):
        self.device = device
        self.batch_blocks = batch_blocks
        self.configure(kernel_id, width)

    def configure(self, kernel_id, width):
        self.kernel_id = kernel_id
        self.width = width
        self.blocks = []
        self.index = {}
        self.chunks = []
        self.outputs = None

    def process(self, files, kernel, kernel_id, width=None, events=None):
        import torch
        width = self.width if width is None else width
        if (kernel_id, width) != (self.kernel_id, self.width):
            self.configure(kernel_id, width)
        planning_start = time.perf_counter()
        first = len(self.blocks)
        first_chunk = len(self.chunks)
        plan = self.plan_type(width=width)
        plan.blocks = self.blocks
        plan.index = self.index
        if isinstance(plan, ChunkBlockPlan):
            plan.chunks = self.chunks
        plan.add('current', files)
        planning = time.perf_counter()-planning_start
        missing = MissingChunkPlan(plan, first) if isinstance(plan, ChunkBlockPlan) else BlockPlan(width=width, reuse=False)
        if not isinstance(plan, ChunkBlockPlan):
            missing.blocks = self.blocks[first:]
            missing.refs = list(range(len(missing.blocks)))
        with torch.inference_mode():
            if missing.blocks:
                new = execute_torch(missing, kernel, self.device, self.batch_blocks, events)
                self.outputs = new if self.outputs is None else torch.cat((self.outputs, new))
            if self.outputs is None:
                raise ValueError('GPU cache requires nonempty input')
            identity = len(plan.refs) == len(self.blocks) and all(i == ref for i, ref in enumerate(plan.refs))
            output = self.outputs if identity else self.outputs.index_select(
                0, torch.tensor(plan.refs, dtype=torch.int64, device=self.device))
            host = output.cpu().numpy()
        restored = plan.reconstruct()['current']
        return host, restored, dict(new_blocks=len(missing.blocks), output_blocks=len(plan.refs),
                                    cached_blocks=len(self.blocks),planning_lookup_s=planning,
                                    chunk_lookup_s=getattr(plan,'lookup_s',0),
                                    new_chunks=len(self.chunks)-first_chunk)


class ChunkBlockPlan(BlockPlan):
    """Index entire content chunks, splitting only verified new chunks."""
    def __init__(self, width=2048, reuse=True):
        super().__init__(width,reuse)
        self.chunks = []
        self.lookup_s = 0

    def add(self, name, files):
        if any(v['name']==name for v in self.versions):
            raise ValueError('Duplicate version')
        entries=[]
        for path,raw in files.items():
            chunk_refs=[]
            self.input_bytes+=len(raw)
            view=memoryview(raw)
            for start,stop in spans(raw):
                part=view[start:stop]
                lookup_start=time.perf_counter()
                digest=hashlib.sha256(part).digest()
                node=self.index.get(digest)
                if node is not None and self.chunks[node]['raw']!=bytes(part):
                    raise ValueError('Hash collision')
                self.lookup_s+=time.perf_counter()-lookup_start
                if node is None:
                    first=len(self.blocks)
                    # Identical offsets and tail lengths to BlockPlan.
                    self.blocks.extend(view[offset:min(offset+self.width,stop)]
                                       for offset in range(start,stop,self.width))
                    node=len(self.chunks)
                    self.chunks.append(dict(raw=bytes(part),first=first,stop=len(self.blocks),
                                            refs=tuple(range(first,len(self.blocks)))))
                    self.index[digest]=node
                chunk=self.chunks[node]
                self.refs.extend(chunk['refs'])
                chunk_refs.append(node)
            entries.append(dict(name=path,chunks=chunk_refs,size=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest()))
        self.versions.append(dict(name=name,files=entries))

    def inputs(self, first=0, stop=None, buffer=None):
        """Pack contiguous full rows in native array copies, one per chunk."""
        stop=len(self.blocks) if stop is None else min(stop,len(self.blocks))
        count=stop-first
        if count<=0:return np.empty((0,self.width),dtype=np.uint8)
        packed=np.empty((count,self.width),dtype=np.uint8) if buffer is None else buffer[:count]
        packed.fill(0)
        # Chunk ranges are append-only and ordered. Locate the first intersecting
        # group without scanning all earlier cached chunks on each update batch.
        node=bisect.bisect_right(self.chunks,first,key=lambda c:c['first'])-1
        for chunk in self.chunks[max(node,0):]:
            if chunk['first']>=stop:break
            begin=max(first,chunk['first']);end=min(stop,chunk['stop'])
            full=len(chunk['raw'])//self.width
            full_end=min(end,chunk['first']+full)
            if begin<full_end:
                rows=np.frombuffer(chunk['raw'],dtype=np.uint8,count=full*self.width).reshape(full,self.width)
                packed[begin-first:full_end-first]=rows[begin-chunk['first']:full_end-chunk['first']]
            if end>chunk['first']+full:
                tail=np.frombuffer(chunk['raw'],dtype=np.uint8,offset=full*self.width)
                packed[chunk['first']+full-first,:len(tail)]=tail
        return packed

    def reconstruct(self):
        result={}
        for version in self.versions:
            files={}
            for entry in version['files']:
                raw=b''.join(self.chunks[node]['raw'] for node in entry['chunks'])
                if len(raw)!=entry['size'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
                    raise ValueError('Reconstruction integrity failed')
                files[entry['name']]=raw
            result[version['name']]=files
        return result


class ChunkGPUCache(BlockGPUCache):
    plan_type = ChunkBlockPlan


class MissingChunkPlan(BlockPlan):
    """Expose only new rows while keeping the parent's chunk-wise packer."""
    def __init__(self,parent,first):
        super().__init__(parent.width,reuse=False)
        self.parent=parent
        self.first=first
        self.blocks=parent.blocks[first:]
        self.refs=list(range(len(self.blocks)))

    def inputs(self,first=0,stop=None,buffer=None):
        end=len(self.blocks) if stop is None else min(stop,len(self.blocks))
        return self.parent.inputs(self.first+first,self.first+end,buffer)


# Public persistent cache now uses one verified lookup per content chunk.
# BlockGPUCache remains available for diagnostic comparisons.
GPUCache = ChunkGPUCache
