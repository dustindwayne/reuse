"""Exercise cache rollback without requiring the optional Torch installation."""
import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from structural_residual_lm import gpu_reuse


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __len__(self):
        return len(self.values)

    def index_select(self, dim, refs):
        return Tensor(self.values[refs])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class CacheFailureTests(unittest.TestCase):
    def test_real_torch_failure_and_retry(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch backend unavailable')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        kernel = lambda rows: rows.to(torch.int64).sum(dim=1)

        def fail(rows):
            raise RuntimeError('Intentional kernel failure')

        for cache_type in (gpu_reuse.BlockGPUCache, gpu_reuse.ChunkGPUCache):
            with self.subTest(cache=cache_type.__name__, device=device):
                cache = cache_type('sum', width=256, device=device)
                cache.process({'file': b'A' * 700}, kernel, 'sum')
                original = (len(cache.blocks), dict(cache.index), len(cache.chunks), cache.outputs)
                for failing in (fail, lambda rows: kernel(rows)[:0]):
                    with self.assertRaises((RuntimeError, ValueError)):
                        cache.process({'file': b'B' * 1100}, failing, 'sum')
                    self.assertEqual((len(cache.blocks), cache.index, len(cache.chunks)), original[:3])
                    self.assertIs(cache.outputs, original[3])
                cache.process({'file': b'C' * 900}, kernel, 'sum')
                for raw in (b'B' * 1100, b'A' * 700, b'C' * 900):
                    actual, restored, _ = cache.process({'file': raw}, kernel, 'sum')
                    dense = gpu_reuse.BlockPlan(width=256, reuse=False)
                    dense.add('v', {'file': raw})
                    np.testing.assert_array_equal(actual, gpu_reuse.execute_numpy(
                        dense, lambda rows: rows.astype(np.int64).sum(axis=1)))
                    self.assertEqual(restored, {'file': raw})

    def test_failed_kernel_does_not_publish_definitions_or_shift_outputs(self):
        torch = types.SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cat=lambda values: Tensor(np.concatenate([v.values for v in values])),
            tensor=lambda refs, **kwargs: np.asarray(refs),
            int64=np.int64,
        )

        def execute(plan, kernel, *args):
            result = kernel(plan.inputs())
            if len(result) != len(plan.blocks):
                raise ValueError('Bad output count')
            return Tensor(result)

        def kernel(rows):
            return rows.astype(np.int64).sum(axis=1)

        def fail(rows):
            raise RuntimeError('Simulated kernel failure')

        with patch.dict(sys.modules, torch=torch), patch.object(gpu_reuse, 'execute_torch', execute):
            for cache_type in (gpu_reuse.BlockGPUCache, gpu_reuse.ChunkGPUCache):
                with self.subTest(cache=cache_type.__name__):
                    cache = cache_type('sum', width=4, device='cpu')
                    cache.process({'file': b'abcd'}, kernel, 'sum')
                    before = (len(cache.blocks), dict(cache.index), len(cache.chunks), cache.outputs)
                    for failing in (fail, lambda rows: rows[:0]):
                        with self.assertRaises((RuntimeError, ValueError)):
                            cache.process({'file': b'efghijkl'}, failing, 'sum')
                        self.assertEqual((len(cache.blocks), cache.index, len(cache.chunks)), before[:3])
                        self.assertIs(cache.outputs, before[3])
                    cache.process({'file': b'mnopqrst'}, kernel, 'sum')
                    for raw in (b'efghijkl', b'abcd', b'mnopqrst'):
                        actual, restored, _ = cache.process({'file': raw}, kernel, 'sum')
                        dense = gpu_reuse.BlockPlan(width=4, reuse=False)
                        dense.add('v', {'file': raw})
                        np.testing.assert_array_equal(actual, gpu_reuse.execute_numpy(dense, kernel))
                        self.assertEqual(restored, {'file': raw})
