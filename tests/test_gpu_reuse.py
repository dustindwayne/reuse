import unittest
import numpy as np
from structural_residual_lm.gpu_reuse import BlockPlan, execute_numpy, GPUCache, ChunkBlockPlan, ChunkGPUCache


class GPUReusePlanTests(unittest.TestCase):
    def test_chunk_groups_match_dense_boundaries_padding_and_order(self):
        versions=self.workload()
        versions['tails']={'short':b'A','long':b'A\0','again':versions['original']['x']}
        chunk=ChunkBlockPlan(width=256)
        dense=BlockPlan(width=256,reuse=False)
        for name,files in versions.items():
            chunk.add(name,files);dense.add(name,files)
        np.testing.assert_array_equal(chunk.inputs()[chunk.refs],dense.inputs())
        packed=chunk.inputs()
        # Transfers can bisect chunk groups and end on padded tails.
        for first in range(0,len(chunk.blocks),37):
            stop=min(first+37,len(chunk.blocks))
            buffer=np.full((37,256),255,dtype=np.uint8)
            np.testing.assert_array_equal(chunk.inputs(first,stop,buffer),packed[first:stop])
        self.assertEqual(chunk.reconstruct(),versions)
        self.assertLess(len(chunk.blocks),len(dense.blocks))
        chunk.chunks[0]['raw']=b'corrupt'
        with self.assertRaises(ValueError):chunk.reconstruct()

    def test_chunk_collision_rejected(self):
        from unittest.mock import patch
        chunk=ChunkBlockPlan(width=256)
        chunk.add('a',{'x':b'abcd'})
        digest=next(iter(chunk.index))
        class Collision:
            def digest(self):return digest
        with patch('structural_residual_lm.gpu_reuse.hashlib.sha256',return_value=Collision()):
            with self.assertRaises(ValueError):chunk.add('b',{'x':b'efgh'})

    def test_chunk_cache_hits_and_invalidation(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch backend unavailable')
        cache=ChunkGPUCache('sum',width=256,device='cpu')
        files=self.workload()['original']
        kernel=lambda rows:rows.to(torch.int64).sum(dim=1)
        first,restored,stats=cache.process(files,kernel,'sum')
        second,restored,stats=cache.process(files,kernel,'sum')
        self.assertEqual(stats['new_blocks'],0)
        self.assertEqual(stats['new_chunks'],0)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(restored,files)
        changed,restored,stats=cache.process(files,lambda rows:kernel(rows)+1,'sum-plus-one')
        np.testing.assert_array_equal(changed,first+1)
        self.assertEqual(stats['new_blocks'],stats['cached_blocks'])
        resized,restored,stats=cache.process(files,kernel,'sum',width=128)
        dense=BlockPlan(width=128,reuse=False);dense.add('v',files)
        np.testing.assert_array_equal(resized,execute_numpy(dense,lambda rows:rows.astype(np.int64).sum(axis=1)))
        self.assertEqual(restored,files)

    def test_persistent_cache_updates_and_configuration_invalidation(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch backend unavailable')
        cache=GPUCache('sum',width=256,device='cpu')
        kernel=lambda rows:rows.to(torch.int64).sum(dim=1)
        versions=self.workload()
        for files in versions.values():
            output,restored,stats=cache.process(files,kernel,'sum')
            dense=self.plan({'current':files},False)
            expected=execute_numpy(dense,lambda rows:rows.astype(np.int64).sum(axis=1))
            np.testing.assert_array_equal(output,expected)
            self.assertEqual(restored,files)
        self.assertLess(stats['new_blocks'],stats['output_blocks'])
        changed,restored,stats=cache.process(files,lambda rows:kernel(rows)+1,'sum-plus-one')
        np.testing.assert_array_equal(changed,expected+1)
        self.assertEqual(stats['new_blocks'],stats['cached_blocks'])

    def workload(self):
        raw=np.random.default_rng(19).integers(0,256,120000,dtype=np.uint8).tobytes()
        return {'original':{'x':raw,'empty':b''},
                'edited':{'x':raw[:33333]+b'inserted'+raw[33333:],'empty':b''}}

    def plan(self, versions, reuse):
        plan=BlockPlan(width=256,reuse=reuse)
        for name,files in versions.items():plan.add(name,files)
        return plan

    def test_full_data_and_complete_outputs_exact(self):
        versions=self.workload();cached=self.plan(versions,True);dense=self.plan(versions,False)
        kernel=lambda x:np.fft.rfft(x.astype(np.float32),axis=1)
        np.testing.assert_array_equal(execute_numpy(cached,kernel),execute_numpy(dense,kernel))
        self.assertEqual(cached.reconstruct(),versions)
        self.assertEqual(dense.reconstruct(),versions)
        self.assertGreater(cached.stats()['reused_bytes'],0)
        self.assertLess(len(cached.blocks),len(dense.blocks))
        self.assertEqual(len(cached.refs),len(dense.refs))

    def test_context_changes_do_not_reuse_outputs(self):
        plan=self.plan(self.workload(),True)
        a=execute_numpy(plan,lambda x:x.astype(np.int64).sum(axis=1))
        b=execute_numpy(plan,lambda x:x.astype(np.int64).sum(axis=1)+1)
        np.testing.assert_array_equal(b,a+1)

    def test_empty_tail_and_zero_padding_are_distinct(self):
        files={'v':{'a':b'A','b':b'A\0','empty':b''}}
        plan=self.plan(files,True)
        self.assertEqual(plan.reconstruct(),files)
        self.assertEqual(len(plan.blocks),2)
        self.assertEqual(plan.lengths,[1,2])

    def test_integrity_and_kernel_contract(self):
        plan=self.plan({'v':{'a':b'data'}},True)
        with self.assertRaises(ValueError):execute_numpy(plan,lambda x:x[:0])
        with self.assertRaises(ValueError):plan.add('v',{'a':b'data'})
        plan.blocks[0]=b'nope'
        with self.assertRaises(ValueError):plan.reconstruct()
        with self.assertRaises(ValueError):BlockPlan(width=0)
