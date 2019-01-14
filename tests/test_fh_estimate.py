from unittest import main, TestCase

import numpy as np
from scipy.optimize import minimize

from pygom import SquareLoss
from pygom.model import common_models


class TestFHEstimate(TestCase):

    def setUp(self):
        # initial values
        x0 = [-1.0, 1.0]
        # params
        param_eval = [('a', 0.2), ('b', 0.2), ('c', 3.0)]
        self.target = np.array([0.2, 0.2, 3.0])
        # the time points for our observations
        t = np.linspace(0, 20, 30).astype('float64')
        ode = common_models.FitzHugh(param_eval)
        ode.initial_values = (x0, t[0])
        solution = ode.integrate(t[1::])
        self.theta = np.array([0.5, 0.5, 0.5])

        self.obj = SquareLoss(self.theta, ode, x0, t[0],
                              t[1::], solution[1::, :], ['V', 'R'])

        g = self.obj.gradient()
        assert np.linalg.norm(g) > 0

        EPSILON = np.sqrt(np.finfo(np.float).eps)

        self.box_bounds = [(EPSILON, 5.0)]*len(self.theta)

    def test_FH_sensitivity(self):
        res = minimize(fun=self.obj.cost,
                       jac=self.obj.sensitivity,
                       x0=self.theta,
                       bounds=self.box_bounds,
                       method='L-BFGS-B')

        self.assertTrue(np.allclose(self.target, res['x'], 1e-2, 1e-2))

    def test_FH_adjoint(self):
        res = minimize(fun=self.obj.cost,
                       jac=self.obj.adjoint,
                       x0=self.theta,
                       bounds=self.box_bounds,
                       method='L-BFGS-B')

        self.assertTrue(np.allclose(self.target, res['x'], 1e-2, 1e-2))

    def test_FH_IV(self):
        box_bounds = self.box_bounds + [(None, None)]*2

        res = minimize(fun=self.obj.costIV,
                       jac=self.obj.sensitivityIV,
                       x0=self.theta.tolist() + [-0.5, 0.5],
                       bounds=box_bounds,
                       method='L-BFGS-B')

        target = np.array([0.2, 0.2, 3.0, -1.0, 1.0])
        self.assertTrue(np.allclose(res['x'], target, 1e-2, 1e-2))


if __name__ == '__main__':
    main()
