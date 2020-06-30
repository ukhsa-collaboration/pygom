from unittest import main, TestCase

import numpy as np


from pygom import Transition, TransitionType, SimulateOde, SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss
from pygom.model import common_models

class Test_Square_loss_class(TestCase):

    def setUp(self):
        # initial values
        N = 1e6
        in_inf = 1
        self.init_state = [N - in_inf, in_inf, 0.0]
        # params
        self.param_eval = [('beta', 3.6), ('gamma', 0.2), ('N', N)]
        # the time points for our observations
        self.t = np.arange (0 , 51 , 0.25)
        states = ['S', 'I', 'R']
        params = ['beta', 'gamma', 'N']
        transitions = [Transition(origin='S', destination='I', equation='beta*S*I/N', 
                                  transition_type=TransitionType.T),
                       Transition(origin='I', destination='R', equation='gamma*I', 
                                  transition_type=TransitionType.T)]
        self.ode = SimulateOde(states, params, transition=transitions)
        self.ode.parameters = self.param_eval
        self.ode.initial_values = (self.init_state, self.t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        self.solution = self.ode.integrate(self.t[1:])
        # initial guess
        self.theta = [3, 0.15,N]

        obj = SquareLoss(self.theta, self.ode, self.init_state, self.t[0],
                         self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        self.r = obj.residual()

    def test_FH_scalar_weights_for_two_states(self):
        # weight for each component
        w = [2.0, 3.0]
        
        s = 0
        for i in range(2): s += ((self.r[:,i]*w[i])**2).sum()

        obj = SquareLoss(self.theta, self.ode, self.init_state, self.t[0],
                         self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)

        self.assertTrue(np.allclose(obj.cost(), s))

    def test_FH_vector_weights_for_two_states(self):
        # now the weight is a vector
        w = np.random.rand(self.solution[1::,1:3].shape[0],self.solution[1::,1:3].shape[1])
        obj = SquareLoss(self.theta, self.ode, self.init_state, self.t[0],                         
                         self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)

        s = ((self.r * np.array(w))**2).sum()

        self.assertTrue(np.allclose(obj.cost(), s))


    def test_FH_1State_weights_Failures_TypeErrors(self):
        w_list = list()
        w_list.append('a')
        test_weight=np.ones(len(self.solution[1::,-1])).tolist()
        test_weight[-1]='b'
        w_list.append(test_weight)
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(TypeError, SquareLoss, self.theta, self.ode,
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
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
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                  'R', w)
        
    def test_FH_1State_weights_Failures_ShapeErrors(self):
        w_list = list()         
        w_list.append([2.0, 3.0])
        w_list.append(np.random.rand(self.solution[1::,-1].shape[0]+1).tolist())
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(AssertionError, SquareLoss, self.theta, self.ode,
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                  'R', w)

    def test_FH_2State_weights_Failures_TypeErrors(self):
        w_list = list()
        w_list.append(['a','b'])
        test_weight=np.ones(self.solution[1::,1:3].shape).tolist()
        test_weight[-1][-1]='c'
        w_list.append(test_weight)
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(TypeError, SquareLoss, self.theta, self.ode,
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,1:3],
                                  ['I', 'R'], w)
        
    def test_FH_2State_weights_Failures_ValueErrors(self):
        w_list = list()
        test_weight=np.ones(self.solution[1::,1:3].shape)       
        w_list.append(-test_weight)
        test_weight[-1,-1]=-1
        w_list.append(test_weight)
        w_list.append(np.zeros(self.solution[1::,1:3].shape))
        w_list.append([1.0, -1.0])
        w_list.append([0, 0])
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(ValueError, SquareLoss, self.theta, self.ode,
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,1:3],
                                  ['I', 'R'], w)
        
    def test_FH_2State_weights_Failures_ShapeErrors(self):
        w_list = list()        
        w_list.append([2.0, 3.0, 4.0])
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0]))
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0]+1,self.solution[1::,1:3].shape[1]).tolist())
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0],self.solution[1::,1:3].shape[1]+1).tolist())
        for w in w_list:
            with self.subTest('Weightings',w=w):
                self.assertRaises(AssertionError, SquareLoss, self.theta, self.ode,
                                  self.init_state, self.t[0], self.t[1::], self.solution[1::,1:3],
                                  ['I', 'R'], w)

# class Test_Normal_loss_class(TestCase):



#     def test_FH_Normal(self):
#         objFH = NormalLoss(self.theta, self.ode, self.init_state, self.t[0],
#                            self.t[1::], self.solution[1::,1:3], ['I', 'R'])

#         w = [2.0, 3.0]
#         objFH1 = NormalLoss(self.theta, self.ode, self.init_state, self.t[0],
#                            self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)

#         # now the weight is a vector
#         w = np.random.rand(29, 2)
#         objFH2 = NormalLoss(self.theta, self.ode, self.init_state, self.t[0],
#                            self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)

#         self.assertFalse(np.allclose(objFH.cost(), objFH1.cost()))
#         self.assertFalse(np.allclose(objFH1.cost(), objFH2.cost()))




if __name__ == '__main__':
    main()
