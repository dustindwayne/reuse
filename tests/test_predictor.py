import unittest
import numpy as np
from structural_residual_lm.predictor import StructuralPredictor

class PredictorTests(unittest.TestCase):
    def test_future_cannot_change_features(self):
        p=StructuralPredictor().fit([["<bos>","the","frog","jumps",".","<eos>"]])
        a=[["<bos>","the","frog","jumps"]];b=[["<bos>","the","frog","SECRET"]]
        fa,_=p.features(a,[(0,3)]);fb,_=p.features(b,[(0,3)])
        np.testing.assert_array_equal(fa,fb)
    def test_signal_finite_and_trie_frozen(self):
        p=StructuralPredictor().fit([["<bos>","the","frog"]]);n=p.trie.node_count()
        f,s=p.features([["<bos>","unknown","☃"]],[(0,2)])
        self.assertTrue(np.isfinite(f).all());self.assertEqual(p.trie.node_count(),n)
