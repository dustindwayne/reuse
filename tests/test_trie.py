import unittest
import numpy as np
from structural_residual_lm.trie import ContextTrie

class TrieTests(unittest.TestCase):
    def test_counts_backoff_and_empty(self):
        t=ContextTrie(2)
        self.assertEqual(t.predict([],1)["confidence"],0)
        for _ in range(5): t.update([(0,),(16,)],(8,))
        p=t.predict([(99,),(16,)],1)
        self.assertEqual(p["depth"],1)
        self.assertEqual(p["count"],5)
        np.testing.assert_allclose(p["expected"],[.5])
        self.assertAlmostEqual(p["confidence"],5/7)
        self.assertEqual(t.predict([(99,)],1)["depth"],0)
    def test_depth_zero_and_mixture(self):
        t=ContextTrie(0);t.update([(0,)],(0,));t.update([(16,)],(16,))
        p=t.predict([(16,)],1)
        self.assertEqual(p["depth"],0);self.assertAlmostEqual(p["entropy"],1)
        np.testing.assert_allclose(p["expected"],[.5])
