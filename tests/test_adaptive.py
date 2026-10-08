import random,unittest
from structural_residual_lm.adaptive import encode,decode,block,restore

class AdaptiveTests(unittest.TestCase):
    def test_maximum_window_header_boundary(self):
        raw=random.Random(37).randbytes(65535)
        self.assertEqual(decode(encode(raw,dynamic=False,extract=False,max_window=65534)),raw)
        with self.assertRaises(ValueError):
            encode(raw,max_window=65535)
    def test_exact(self):
        rng=random.Random(3)
        for raw in (b'',bytes(range(256)),rng.randbytes(3000),b's'*3000,bytes([0,255])*1000,'Case\r\n café𐍈'.encode()):
            for dynamic in (False,True):
                for extract in (False,True):
                    packed=encode(raw,dynamic,extract)
                    self.assertEqual(decode(packed),raw)
                    self.assertLessEqual(len(packed),len(raw)+5)
    def test_extraction_and_split(self):
        raw=bytes([0,255])*512
        self.assertEqual(block(raw)[0],2)
        self.assertEqual(restore(block(raw),len(raw)),raw)
        self.assertLess(len(encode(raw,extract=True)),len(encode(raw,extract=False)))
        raw=bytes(range(16))*32+bytes(range(240,256))*32
        self.assertLess(len(encode(raw,dynamic=True,extract=False)),len(encode(raw,dynamic=False,extract=False)))
    def test_bad_stream(self):
        good=encode(b's'*2000)
        for bad in (b'',good[:-1],good+b'x',b'BAD!'+good[4:]):
            with self.assertRaises(ValueError): decode(bad)
