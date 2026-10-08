import unittest
import numpy as np
from structural_residual_lm.residual import residual

class ResidualTests(unittest.TestCase):
    def test_signed_reconstruction(self):
        a=np.array([0.,1.,.25]);p=np.array([.5,.2,.9]);r=residual(a,p)
        np.testing.assert_allclose(p+r,a);self.assertLess(r[0],0)
    def test_shape(self):
        with self.assertRaises(ValueError): residual([1],[1,2])
