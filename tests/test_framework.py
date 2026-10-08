import random,unittest
from structural_residual_lm.framework import pack,unpack,filewise
from structural_residual_lm.patterns import pack as pattern
class FrameworkTests(unittest.TestCase):
    def test_full_path_selection(self):
        rng=random.Random(5);base=rng.randbytes(2048)
        files={'a':base,'b':b'insert'+base,'empty':b''}
        candidates=[pattern(files,matching=False),pattern(files),filewise(files)]
        chosen=pack(files)
        self.assertEqual(len(chosen),min(map(len,candidates)))
        for blob in candidates+[chosen]:self.assertEqual(unpack(blob),files)
    def test_empty_and_malformed(self):
        self.assertEqual(unpack(pack({})),{})
        for blob in (b'',b'FAW1\x02hello',b'FAW1\x01not-zlib'):
            with self.assertRaises(ValueError):unpack(blob)
