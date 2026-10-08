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

    def test_random_controls_are_causal(self):
        from structural_residual_lm.experiment import feature_split
        p=StructuralPredictor().fit([["<bos>","the","frog","jumps"]])
        cfg=dict(system="residual",seed=7,random_surprise=True,random_structure=True)
        a=[["<bos>","the","frog","jumps"]];b=[["<bos>","the","frog","SECRET"]]
        f,l,s=feature_split(p,a,[(0,3)],cfg);g,m,t=feature_split(p,b,[(0,3)],cfg)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(l,m);np.testing.assert_array_equal(s["surprise"],t["surprise"])
