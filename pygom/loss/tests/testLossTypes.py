from unittest import TestCase

import numpy as np

from pygom import SquareLoss, NormalLoss
from pygom.model import common_models

class TestLossTypes(TestCase):

    def test_FH_Square(self):
        # initial values
        x0 = [-1.0, 1.0]
        # params
        param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]
        # the time points for our observations
        t = np.linspace(0, 20, 30).astype('float64')
        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        solution, _output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        #objFH = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        objFH = SquareLoss(theta, ode, x0, t[0], t[1::],
                           solution[1::,:], ['V','R'])

        r = objFH.residual()

        # weight for each component
        w = [2.0, 3.0]

        s1 = 0
        for i in range(2): s1 += ((r[:,i]*w[i])**2).sum()

        objFH1 = SquareLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,:], ['V','R'], w)

        # now the weight is a vector
        w = np.random.rand(29, 2)
        objFH2 = SquareLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,:], ['V','R'], w)

        s2 = ((r * np.array(w))**2).sum()

        self.assertTrue(np.allclose(objFH1.cost(), s1))
        self.assertTrue(np.allclose(objFH2.cost(), s2))

    def test_FH_Normal(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t0)
        # the time points for our observations
        t = np.linspace(0, 20, 30).astype('float64')
        # Standard.  Find the solution which we will be used as "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        #objFH = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        objFH = NormalLoss(theta, ode, x0, t[0], t[1::],
                           solution[1::,:], ['V','R'])

        w = [2.0,3.0]
        objFH1 = NormalLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,:], ['V','R'], w)

        # now the weight is a vector
        w = np.random.rand(29, 2)
        objFH2 = NormalLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,:], ['V','R'], w)

        objFH.cost()
        objFH1.cost()
        objFH2.cost()

    def test_FH_Square_1State_Fail(self):
        ## totalFail = 0
        ## expectedFail = 4
        # initial values
        x0 = [-1.0, 1.0]
        # the time points for our observations
        t = np.linspace(0, 20, 30).astype('float64')
        # params
        param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        w_list = list()

        w_list.append([-1.])
        w_list.append([0])
        w_list.append([2.0, 3.0])
        w_list.append(np.random.rand(30))

        for w in w_list:
            self.assertRaises(AssertionError, SquareLoss, theta, ode, x0,
                              t[0], t[1::], solution[1::,:], 'R', w)

    def test_FH_Square_2State_Fail(self):
        ## totalFail = 0
        ## expectedFail = 8
        # initial values
        x0 = [-1.0, 1.0]
        # the time points for our observations
        t = np.linspace(0, 20, 30).astype('float64')
        # params
        param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        solution, _output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        w_list = list()

        w_list.append([-2.0])
        w_list.append([2.0, 3.0, 4.0])
        w_list.append([0.0, 0.0])
        w_list.append([1.0, -1.0])
        w_list.append(np.random.rand(30))
        w_list.append([np.random.rand(30), np.random.rand(31)])
        w_list.append([np.random.rand(31), np.random.rand(31)])
        w_list.append([np.random.rand(30), np.random.rand(30), np.random.rand(30)])

        for w in w_list:
            self.assertRaises(AssertionError, SquareLoss, theta, ode, x0,
                             t[0], t[1::], solution[1::,:], 'R', w)
