from unittest import main, TestCase

import numpy as np

from pygom import SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss
from pygom.model import common_models

class Test_Square_loss_class(TestCase):

    def setUp(self):
        # initial values
        self.x0 = [-1.0, 1.0]
        # params
        self.param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]
        # the time points for our observations
        self.t = np.linspace(0, 20, 30).astype('float64')
        self.ode = common_models.FitzHugh(self.param_eval)
        self.ode.initial_values = (self.x0, self.t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        self.solution = self.ode.integrate(self.t[1::])
        # initial guess
        self.theta = [0.5, 0.5, 0.5]

        obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],
                         self.t[1::], self.solution[1::,:], ['V', 'R'])
        self.r = obj.residual()

    def test_FH_scalar_weights_for_two_states(self):
        # weight for each component
        w = [2.0, 3.0]

        s = 0
        for i in range(2): s += ((self.r[:,i]*w[i])**2).sum()

        obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],
                         self.t[1::], self.solution[1::,:], ['V', 'R'], w)

        self.assertTrue(np.allclose(obj.cost(), s))

    def test_FH_vector_weights_for_two_states(self):
        # now the weight is a vector
        w = np.random.rand(29, 2)
        obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],                         
                         self.t[1::], self.solution[1::,:], ['V', 'R'], w)

        s = ((self.r * np.array(w))**2).sum()

        self.assertTrue(np.allclose(obj.cost(), s))


    def test_FH_1State_weights_Failures_TypeErrors(self):
        w_list = list()
        w_list.append(['a'])
        w_list.append([True])
        test_weight=np.ones(len(self.solution[1::,-1]))
        test_weight[-1]=False
        w_list.append(test_weight)
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(TypeError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,-1],
                                  'R', w)
                
    def test_FH_1State_weights_Failures_ValueErrors(self):
        w_list = list()
        test_weight=np.ones(len(self.solution[1::,-1]))
        w_list.append(-test_weight)
        test_weight[-1]=-1
        w_list.append(test_weight)
        w_list.append(np.zeros(len(self.solution[1::,-1])))
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(ValueError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,-1],
                                  'R', w)
        
    def test_FH_1State_weights_Failures_ShapeErrors(self):
        w_list = list()         
        w_list.append([2.0, 3.0])
        w_list.append(np.random.rand(30))
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(AssertionError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,-1],
                                  'R', w)

    def test_FH_2State_weights_Failures_TypeErrors(self):
        w_list = list()
        w_list.append(['a','b'])
        w_list.append([True,False])
        test_weight=np.ones(self.solution[1::,:].shape)
        test_weight[-1,-1]=False
        w_list.append(test_weight)
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(TypeError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,:],
                                  ['V', 'R'], w)
        
    def test_FH_2State_weights_Failures_ValueErrors(self):
        w_list = list()
        test_weight=np.ones(self.solution[1::,:].shape)       
        w_list.append(-test_weight)
        test_weight[-1,-1]=-1
        w_list.append(test_weight)
        w_list.append(np.zeros(self.solution[1::,:].shape))
        w_list.append([1.0, -1.0])
        w_list.append([0, 0])
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(ValueError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,:],
                                  ['V', 'R'], w)
        
    def test_FH_2State_weights_Failures_ShapeErrors(self):
        w_list = list()        
        w_list.append([2.0, 3.0, 4.0])
        w_list.append(np.random.rand(30))
        w_list.append([np.random.rand(30), np.random.rand(31)])
        w_list.append([np.random.rand(31), np.random.rand(31)])
        w_list.append([np.random.rand(30), np.random.rand(30), np.random.rand(30)])
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(AssertionError, SquareLoss, self.theta, self.ode,
                                  self.x0, self.t[0], self.t[1::], self.solution[1::,:],
                                  ['V', 'R'], w)

# class Test_Normal_loss_class(TestCase):

#     def setUp(self):
#         # initial values
#         self.x0 = [-1.0, 1.0]
#         # params
#         self.param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]
#         # the time points for our observations
#         self.t = np.linspace(0, 20, 30).astype('float64')
#         self.ode = common_models.FitzHugh(self.param_eval)
#         self.ode.initial_values = (self.x0, self.t[0])

#         # Standard.  Find the solution which we will be used as
#         # "observations later"
#         self.solution = self.ode.integrate(self.t[1::])
#         # initial guess
#         self.theta = [0.5, 0.5, 0.5]

#         obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],
#                          self.t[1::], self.solution[1::,:], ['V', 'R'])
#         self.r = obj.residual()

#     def test_FH_Normal_scalar_weight(self):
#         # weight for each component
#         w = [2.0, 3.0]

#         s = 0
#         for i in range(2): s += ((self.r[:,i]*w[i])**2).sum()

#         obj = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
#                          self.t[1::], self.solution[1::,:], ['V', 'R'], w)

#         self.assertTrue(np.allclose(obj.cost(), s))

#     def test_FH_Normal_vector_weight(self):
#         # now the weight is a vector
#         w = np.random.rand(29, 2)
#         obj = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
#                          self.t[1::], self.solution[1::,:], ['V', 'R'], w)

#         s = ((self.r * np.array(w))**2).sum()

#         self.assertTrue(np.allclose(obj.cost(), s))

#     def test_FH_Normal(self):
#         objFH = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
#                            self.t[1::], self.solution[1::,:], ['V', 'R'])

#         w = [2.0, 3.0]
#         objFH1 = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
#                            self.t[1::], self.solution[1::,:], ['V', 'R'], w)

#         # now the weight is a vector
#         w = np.random.rand(29, 2)
#         objFH2 = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
#                            self.t[1::], self.solution[1::,:], ['V', 'R'], w)

#         self.assertFalse(np.allclose(objFH.cost(), objFH1.cost()))
#         self.assertFalse(np.allclose(objFH1.cost(), objFH2.cost()))

#     def test_FH_Normal_1State_Fail(self):
#         ## totalFail = 0
#         ## expectedFail = 4
#         w_list = list()

#         w_list.append([-1.])
#         w_list.append([0])
#         w_list.append([2.0, 3.0])
#         w_list.append(np.random.rand(30))

#         for w in w_list:
#             self.assertRaises(AssertionError, NormalLoss, self.theta, self.ode,
#                               self.x0, self.t[0], self.t[1::], self.solution[1::,:],
#                               'R', w)

#     def test_FH_Normal_2State_Fail(self):
#         ## totalFail = 0
#         ## expectedFail = 8
#         w_list = list()

#         w_list.append([-2.0])
#         w_list.append([2.0, 3.0, 4.0])
#         w_list.append([0.0, 0.0])
#         w_list.append([1.0, -1.0])
#         w_list.append(np.random.rand(30))
#         w_list.append([np.random.rand(30), np.random.rand(31)])
#         w_list.append([np.random.rand(31), np.random.rand(31)])
#         w_list.append([np.random.rand(30), np.random.rand(30), np.random.rand(30)])

#         for w in w_list:
#             self.assertRaises(AssertionError, NormalLoss, self.theta, self.ode,
#                               self.x0, self.t[0], self.t[1::], self.solution[1::,:],
#                               'R', w)


if __name__ == '__main__':
    main()
