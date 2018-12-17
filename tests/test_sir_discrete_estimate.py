from unittest import main, TestCase

import numpy as np
import scipy.optimize

from pygom import PoissonLoss
from pygom import Transition, TransitionType, DeterministicOde


class TestSIRDiscreteEstimate(TestCase):

    def setUp(self):
        # initial values
        N = 2362205.0
        self.x0 = [N, 3.0, 0.0]
        self.t = np.linspace(0, 150, 100).astype('float64')
        # params
        param_eval = [('beta', 0.5), ('gamma', 1.0/3.0), ('N', N)]

        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta * S * I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma * I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        self.ode = DeterministicOde(state_list, param_list, transition=transition_list)
        self.ode.parameters = param_eval
        self.ode.initial_values = (self.x0, self.t[0])

        # Standard.  Find the solution.
        self.solution = self.ode.integrate(self.t[1::])
        # initial value
        self.theta = np.array([0.4, 0.3])

        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        self.box_bounds = [(EPSILON, 2), (EPSILON, 2)]
        self.target = np.array([0.5, 1.0/3.0])

    def test_SIR_Estimate_PoissonLoss_1TargetState(self):
        obj = PoissonLoss(self.theta, self.ode, self.x0, self.t[0],
                          self.t[1::], np.round(self.solution[1::,2]),
                          'R', target_param=['beta', 'gamma'])

        res = scipy.optimize.minimize(fun=obj.cost,
                                      jac=obj.sensitivity,
                                      x0=self.theta,
                                      method='L-BFGS-B',
                                      bounds=self.box_bounds)

        self.assertTrue(np.allclose(res['x'], self.target))

    def test_SIR_Estimate_PoissonLoss_2TargetState(self):
        # note that we need to round the observations to integer for it
        # to make sense
        obj = PoissonLoss(self.theta, self.ode, self.x0, self.t[0],
                          self.t[1::], np.round(self.solution[1::,1:3]),
                          ['I', 'R'], target_param=['beta', 'gamma'])

        res = scipy.optimize.minimize(fun=obj.cost,
                                      jac=obj.sensitivity,
                                      x0=self.theta,
                                      method='L-BFGS-B',
                                      bounds=self.box_bounds)

        self.assertTrue(np.allclose(res['x'], self.target))


if __name__ == '__main__':
    main()
