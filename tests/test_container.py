import random,unittest
from structural_residual_lm.container import pack,unpack,restore
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

    def test_transform_then_zlib(self):
        import zlib
        from structural_residual_lm.adaptive import encode
        from structural_residual_lm.grid import encode as grid
        from structural_residual_lm.lossless import Grammar
        raw=bytes(range(16))*64+bytes([255])*300
        for mode,transformed in ((5,encode(raw)),(6,Grammar.build(raw).dumps()),(7,grid(raw))):
            self.assertEqual(restore(mode,zlib.compress(transformed)),raw)
