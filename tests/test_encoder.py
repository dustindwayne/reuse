import unittest
import numpy as np
from structural_residual_lm.encoder import StructuralEncoder

class EncoderTests(unittest.TestCase):
    def test_deterministic_and_roundtrip(self):
        e=StructuralEncoder(); restored=StructuralEncoder.loads(e.dumps())
        for token in ["running", "", "☃", "éclair", "123", "!!!"]:
            np.testing.assert_array_equal(e.encode(token),restored.encode(token))
            self.assertTrue(np.isfinite(e.encode(token)).all())
            self.assertTrue(((e.encode(token)>=0)&(e.encode(token)<=1)).all())
    def test_structure_distinguishable(self):
        e=StructuralEncoder()
        self.assertNotEqual(e.state("running"),e.state("123"))
    def test_disabled(self):
        e=StructuralEncoder(disabled=("prefix",))
        self.assertEqual(e.encode("redo")[12:15].sum(),0)
