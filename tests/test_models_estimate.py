from unittest import TestCase

import copy

import numpy as np
import scipy.integrate
import scipy.optimize

from pygom import SquareLoss, NormalLoss, PoissonLoss
from pygom import Transition, TransitionType, DeterministicOde
from pygom.model import common_models

class TestModelEstimate(TestCase):

    def test_SIR_Estimate_SquareLoss(self):
        # define the model and parameters
        ode = common_models.SIR({'beta': 0.5, 'gamma': 1.0/3.0})

        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        # y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        # test out whether the single state function 'ok'
        sir_obj = SquareLoss(theta, ode, x0, t[0], t[1::],
                             solution[1::,2], 'R')
        sir_obj.cost()
        sir_obj.gradient()
        sir_obj.hessian()

        # now we go on the real shit
        sir_obj = SquareLoss(theta, ode, x0, t[0], t[1::],
                             solution[1::,1:3], ['I','R'])

        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [(EPSILON, 5), (EPSILON, 5)]

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                         jac=sir_obj.sensitivity,
                                         x0=theta,
                                         method='SLSQP',
                                         bounds=box_bounds)

        target = np.array([0.5, 1.0/3.0])
        self.assertTrue(np.allclose(res_QP['x'], target, 1e-2, 1e-2))

    def test_SIR_Estimate_SquareLoss_Adjoint(self):
        # define the model and parameters
        ode = common_models.SIR({'beta': 0.5, 'gamma': 1.0/3.0})

        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        sir_obj = SquareLoss(theta, ode, x0, t[0], t[1::], y[1::,:], ['I','R'])

        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [(EPSILON, 5),(EPSILON, 5)]

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                        jac=sir_obj.adjoint,
                                        x0=theta,
                                        method='SLSQP',
                                        bounds=box_bounds)

        target = np.array([0.5, 1.0/3.0])
        self.assertTrue(np.allclose(res_QP['x'], target, 1e-2, 1e-2))

    def test_SIR_Estimate_NormalLoss(self):
        # define the model and parameters
        ode = common_models.SIR({'beta': 0.5, 'gamma': 1.0/3.0})

        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        sir_obj = NormalLoss(theta, ode, x0, t[0], t[1::], y[1::,:], ['I','R'])

        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [(EPSILON, 5), (EPSILON, 5)]

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                        jac=sir_obj.sensitivity,
                                        x0=theta,
                                        method='SLSQP',
                                        bounds=box_bounds)

        target = np.array([0.5, 1.0/3.0])
        self.assertTrue(np.allclose(res_QP['x'], target, 1e-2, 1e-2))

    def test_SIR_Estimate_PoissonLoss_1TargetState(self):
        # initial values
        N = 2362205.0
        x0 = [N, 3.0, 0.0]
        t = np.linspace(0, 150, 100).astype('float64')
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
        ode = DeterministicOde(state_list, param_list, transition=transition_list)
        ode.parameters = param_eval
        ode.initial_values = (x0, t[0])

        # Standard.  Find the solution.
        solution = ode.integrate(t[1::])
        # initial value
        theta = [0.4, 0.3]

        sir_obj = PoissonLoss(theta, ode, x0, t[0], t[1::],
                              np.round(solution[1::,2]),
                              'R', target_param=['beta', 'gamma'])


        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [(EPSILON, 2), (EPSILON, 2)]

        res = scipy.optimize.minimize(fun=sir_obj.cost,
                                      jac=sir_obj.sensitivity,
                                      x0=theta,
                                      method='L-BFGS-B',
                                      bounds=box_bounds)

        target = np.array([0.5, 1.0/3.0])
        self.assertTrue(np.allclose(res['x'], target))

    def test_SIR_Estimate_PoissonLoss_2TargetState(self):
        # initial values
        N = 2362205.0
        x0 = [N, 3.0, 0.0]
        t = np.linspace(0, 150, 100).astype('float64')
        # params
        param_eval = [('beta', 0.5), ('gamma', 1.0/3.0),('N', N)]

        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        ode = DeterministicOde(state_list, param_list,
                               transition=transition_list)
        ode.parameters = param_eval
        ode.initial_values = (x0, t[0])

        # Standard.  Find the solution.
        solution = ode.integrate(t[1::])
        # initial value
        theta = [0.4,0.3]

        # note that we need to round the observations to integer for it
        # to make sense
        objSIR = PoissonLoss(theta, ode, x0, t[0], t[1::],
                             np.round(solution[1::,1:3]),
                             ['I', 'R'], target_param=['beta', 'gamma'])

        # constraints
        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [(EPSILON, 2), (EPSILON, 2)]

        res = scipy.optimize.minimize(fun=objSIR.cost,
                                      jac=objSIR.sensitivity,
                                      x0=theta,
                                      method='L-BFGS-B',
                                      bounds=box_bounds)

        target = np.array([0.5, 1.0/3.0])
        self.assertTrue(np.allclose(res['x'], target))

    def test_FH_Obj(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        param_eval = [('a', 0.2), ('b', 0.2), ('c', 3.0)]

        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t0)
        # the time points for our observations
        t = np.linspace(1, 20, 30).astype('float64')
        # Find the solution which we will be used as "observations later"
        solution, _output = ode.integrate(t, full_output=True)
        # initial guess
        theta = [0.5,0.5,0.5]

        #fh_obj = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        fh_obj = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V', 'R'])

        g1 = fh_obj.adjoint(theta)
        #g2 = fh_obj.adjointInterpolate1(theta)
        #g3 = fh_obj.adjointInterpolate2(theta)
        g4 = fh_obj.sensitivity(theta)

        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (EPSILON, 5.0)
            ]

        res = scipy.optimize.minimize(fun=fh_obj.cost,
                                      jac=fh_obj.sensitivity,
                                      x0=theta,
                                      bounds=box_bounds,
                                      method='L-BFGS-B')

        res2 = scipy.optimize.minimize(fun=fh_obj.cost,
                                       jac=fh_obj.adjoint,
                                       x0=theta,
                                       bounds=box_bounds,
                                       method='L-BFGS-B')

        target = np.array([0.2, 0.2, 3.0])
        self.assertTrue(np.allclose(target, res['x'], 1e-2, 1e-2))
        self.assertTrue(np.allclose(target, res2['x'], 1e-2, 1e-2))

    def test_FH_IV(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        param_eval = [('a', 0.2), ('b', 0.2), ('c', 3.0)]

        ode = common_models.FitzHugh()
        ode.parameters =  param_eval
        ode.initial_values = (x0, t0)
        # the time points for our observations
        t = np.linspace(1, 20, 30).astype('float64')
        # Standard.  Find the solution.
        solution, _output = ode.integrate(t, full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        fh_obj = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V','R'])

        EPSILON = np.sqrt(np.finfo(np.float).eps)

        box_bounds = [
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (None, None),
            (None, None)
            ]

        res = scipy.optimize.minimize(fun=fh_obj.costIV,
                                      jac=fh_obj.sensitivityIV,
                                      x0=theta + [-0.5,0.5],
                                      bounds=box_bounds,
                                      method='L-BFGS-B')

        target = np.array([0.2, 0.2, 3.0, -1.0, 1.0])
        self.assertTrue(np.allclose(res['x'], target, 1e-2, 1e-2))
