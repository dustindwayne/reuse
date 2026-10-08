import random,unittest
from structural_residual_lm.container import pack,unpack
class ContainerTests(unittest.TestCase):
    def test_shared_exact(self):
        rng=random.Random(4);shared=rng.randbytes(4096)
        files={'α.bin':shared*2,'same.bin':shared,'empty':b'','unicode.txt':'Hi\r\n𐍈'.encode()}
        blob=pack(files);out,stats=unpack(blob,True)
        self.assertEqual(out,files);self.assertEqual(stats['total_references'],4)
        self.assertEqual(stats['unique_chunks'],2)
        self.assertLess(len(blob),sum(map(len,files.values())))
    def test_corruption_and_empty(self):
        self.assertEqual(unpack(pack({})),{})
        good=pack({'x':b'a'*1000})
        for bad in (good[:-1],good+b'x',good[:-1]+bytes([good[-1]^1])):
            with self.assertRaises((ValueError,)):unpack(bad)
