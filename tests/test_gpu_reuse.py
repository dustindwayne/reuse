import unittest
import numpy as np
from structural_residual_lm.gpu_reuse import BlockPlan, execute_numpy


class GPUReusePlanTests(unittest.TestCase):
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
