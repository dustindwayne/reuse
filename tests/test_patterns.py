import random,unittest
from structural_residual_lm.patterns import pack,unpack,factor
class PatternTests(unittest.TestCase):
    def test_shifted_and_edits(self):
        rng=random.Random(5);base=rng.randbytes(8192)
        files={'original':base,'shifted':b'INSERT'+base,'edited':base[:3000]+b'changed'+base[3000:]}
        blob=pack(files);decoded,stats=unpack(blob,True)
        self.assertEqual(decoded,files);self.assertGreater(stats['referenced_bytes'],len(base))
        self.assertLess(len(blob),len(pack(files,matching=False)))
    def test_arbitrary_and_empty(self):
        rng=random.Random(3)
        for raw in (b'',b'a'*3000,bytes(range(256))*12,rng.randbytes(2000),'Café\r\n𐍈'.encode()):
            self.assertEqual(unpack(pack({'x':raw,'empty':b''})),{'x':raw,'empty':b''})
    def test_collision_and_corruption(self):
        raw=b'0123456789abcdefAAAA'+b'X'*30+b'0123456789abcdefBBBB'
        self.assertEqual(unpack(pack({'x':raw})),{'x':raw})
        blob=pack({'x':b'abc'*1000})
        for bad in (blob[:-1],b'BAD!'+blob[4:]):
            with self.assertRaises(ValueError):unpack(bad)
