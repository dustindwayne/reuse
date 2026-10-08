import random,unittest,zlib
from structural_residual_lm.folding import fold,unfold,triplet313,untriplet313
from structural_residual_lm.container import select,restore
class FoldingTests(unittest.TestCase):
    def test_exact_widths_and_tails(self):
        rng=random.Random(19)
        for n in list(range(20))+[127,128,129,1000]:
            raw=rng.randbytes(n)
            for w in (2,4,8,16,32,64,128):
                self.assertEqual(unfold(fold(raw,w),w),raw)
                self.assertEqual(restore(8,bytes([w])+zlib.compress(fold(raw,w))),raw)
            self.assertEqual(untriplet313(triplet313(raw)),raw)
            self.assertEqual(restore(9,zlib.compress(triplet313(raw))),raw)
    def test_selection_cannot_increase_payload(self):
        raw=bytes(range(32))*128
        old=select(raw,folding=False);new=select(raw)
        self.assertLessEqual(len(new[1]),len(old[1]));self.assertEqual(restore(*new),raw)
    def test_bad_width(self):
        with self.assertRaises(ValueError):restore(8,b'\0'+zlib.compress(b'hello'))
