import random
import unittest
from structural_residual_lm.lossless import Grammar

class LosslessTests(unittest.TestCase):
    def test_exact_bytes(self):
        rng = random.Random(19)
        cases = [b'', bytes(range(256)), 'S s\t café\r\n𐍈 s!'.encode(), b's'*4096]
        cases += [rng.randbytes(n) for n in (1, 17, 256, 4096)]
        for raw in cases:
            g = Grammar.build(raw)
            restored = Grammar.loads(g.dumps())
            self.assertEqual(restored.decode(), raw)
            self.assertEqual(sum(restored.node_lengths()[r] for r in restored.roots), len(raw))
            self.assertEqual(len(g.terminals), len(set(raw)))
        self.assertLess(len(Grammar.build(b's'*4096).dumps()), 256)

    def test_sharing_and_determinism(self):
        raw = b'the frog jumps. the frog jumps. '*100
        a, b = Grammar.build(raw), Grammar.build(raw)
        self.assertEqual(a.dumps(), b.dumps())
        self.assertEqual(len(a.pairs), len(set(a.pairs)))
        self.assertLess(len(a.dumps()), len(raw))

    def test_invalid_serialization(self):
        for data in (b'', b'LSG1\x80', b'LSG1\x01s\x01\x01\x00\x00', b'LSG1\x00\x00\x01\x00'):
            with self.assertRaises(ValueError):
                Grammar.loads(data)
