import unittest
import numpy as np
from structural_residual_lm.gating import compute_levels
from structural_residual_lm.surprise import surprise

class GatingTests(unittest.TestCase):
    def test_bounds_and_monotonicity(self):
        self.assertEqual(surprise(1,0,0),0);self.assertEqual(surprise(0,1,1),1)
        np.testing.assert_array_equal(compute_levels([0,.249,.25,.49,.50,1]),[0,0,1,1,2,2])
    def test_reject_invalid(self):
        with self.assertRaises(ValueError): surprise(float("nan"),0,0)
        with self.assertRaises(ValueError): compute_levels([.1],(.5,.2))
