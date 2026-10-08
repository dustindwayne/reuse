import random,struct,unittest
from structural_residual_lm.fast import Store,loads,spans,encode_chunk,decode_chunk
class FastTests(unittest.TestCase):
    def test_versions_exact_and_reuse(self):
        raw=random.Random(19).randbytes(200000);files={'a':raw,'empty':b''}
        changed={'a':raw[:33333]+b'insert'+raw[33333:],'empty':b''}
        s=Store();s.add('first',files);s.add('second',changed)
        self.assertEqual(loads(s.dumps()),{'first':files,'second':changed})
        self.assertGreater(s.stats['reused_bytes'],len(raw)//2)
        self.assertLessEqual(s.stats['sample_trials'],4*s.stats['full_encodes'])
    def test_boundaries_cover_and_cap(self):
        raw=b'hello\n\n'*20000;parts=list(spans(raw))
        self.assertEqual(b''.join(raw[a:b] for a,b in parts),raw)
        self.assertTrue(all(b-a<=65536 for a,b in parts))
    def test_corruption(self):
        s=Store();s.add('v',{'x':b'abc'*1000});blob=s.dumps()
        for bad in (blob[:-1],blob+b'x',b'BAD!'+blob[4:]):
            with self.assertRaises(ValueError):loads(bad)
    def test_sampled_folding_is_exact(self):
        raw=b''.join(struct.pack('<I',value) for value in range(16384))
        payload,trials=encode_chunk(raw,adaptive=True)
        self.assertEqual(trials,4)
        self.assertNotEqual(payload[1],0)
        self.assertEqual(decode_chunk(payload),raw)
