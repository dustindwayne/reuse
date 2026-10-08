import tempfile,unittest
from pathlib import Path
from structural_residual_lm.data import prepare,load

class DataTests(unittest.TestCase):
    def test_document_split_and_hashes_repeat(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            m=prepare(a);n=prepare(b)
            self.assertEqual(m["hashes"],n["hashes"])
            train=set((Path(a)/"train.txt").read_text().splitlines());val=set((Path(a)/"validation.txt").read_text().splitlines())
            self.assertFalse(train&val)
            d=load(a);self.assertEqual(len(d["train"][0]),len(d["train"][1]))
            self.assertTrue(all(len(x)==4 for x in d["train"][0]))
    def test_validation_not_in_vocabulary(self):
        with tempfile.TemporaryDirectory() as a:
            (Path(a)/"train.txt").write_text("frog jumps.\n")
            (Path(a)/"validation.txt").write_text("unseenword jumps.\n")
            d=load(a);self.assertNotIn("unseenword",d["vocab"]);self.assertEqual(d["validation"][1][0],0)
