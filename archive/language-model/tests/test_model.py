import unittest
import numpy as np
from structural_residual_lm.model import NeuralLM

class ModelTests(unittest.TestCase):
    def test_manual_gradient_conditional_blocks(self):
        m=NeuralLM(5,context=2,embedding=2,hidden=3,features=2,seed=1)
        x=np.array([[0,1],[2,1],[3,4]]);y=np.array([2,3,0]);f=np.array([[.2,.4],[.3,.1],[.9,.1]]);levels=np.array([0,1,2])
        _,g=m.gradients(x,y,f,levels)
        for name,index in [("E",(1,0)),("W",(0,1)),("W0",(0,0)),("W1",(1,2)),("O",(1,2)),("F",(0,1)),("b0",(1,))]:
            old=m.p[name][index];eps=1e-5
            m.p[name][index]=old+eps;plus=m.gradients(x,y,f,levels)[0]
            m.p[name][index]=old-eps;minus=m.gradients(x,y,f,levels)[0];m.p[name][index]=old
            self.assertAlmostEqual(g[name][index],(plus-minus)/(2*eps),places=6)
    def test_skipped_blocks_have_zero_gradients(self):
        m=NeuralLM(5,context=2,embedding=2,hidden=3)
        _,g=m.gradients(np.array([[0,1]]),np.array([2]),levels=np.array([0]))
        self.assertEqual(g["W0"].sum(),0);self.assertEqual(g["W1"].sum(),0)
