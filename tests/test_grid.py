import random
import unittest
from structural_residual_lm.grid import encode, decode

class GridTests(unittest.TestCase):
    def test_exact_all_shapes(self):
        rng = random.Random(33)
        cases = [b'', bytes([1,11,2,20,0,22,33,21]), bytes(range(256)), b's'*4096]
        cases += [rng.randbytes(n) for n in (1,17,257,1000)]
        for raw in cases:
            for width,bw,bh in ((4,2,2),(17,8,3),(64,8,8),(1,1,1)):
                for adaptive in (False,True):
                    self.assertEqual(decode(encode(raw,width,bw,bh,adaptive)),raw)

    def test_clustered_saves_and_adaptive_fallback(self):
        raw = bytes([127,128,129,130])*1024
        self.assertLess(len(encode(raw)),len(raw))
        irregular = bytes(range(256))*16
        self.assertLessEqual(len(encode(irregular)),len(encode(irregular,adaptive=False)))

    def test_reject_malformed(self):
        packed = encode(b'abc'*100)
        for bad in (b'',packed[:-1],packed+b'x',b'BAD!'+packed[4:]):
            with self.assertRaises(ValueError): decode(bad)
