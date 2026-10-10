from unittest import main, TestCase, skip

import numpy as np
import scipy.optimize

from pygom import SquareLoss, NormalLoss
from pygom.model import common_models


class TestModelEstimate(TestCase):

    def setUp(self):
        # define the model and parameters
        self.beta_true = 0.5
        self.gamma_true = 1.0/3.0
        self.ode = common_models.SIR_norm({'beta': self.beta_true, 'gamma': self.gamma_true})

        # the initial state, normalized to zero one
        i0 = 1.27e-6
        self.x0 = [1-i0, i0, 0]
        # set the time sequence that we would like to observe
        self.t = np.linspace(0, 150, 100)
        self.ode.initial_values = (self.x0, self.t[0])
        # find the solution
        self.solution = self.ode.solve_deterministic(self.t)[0].result.y

        # what the posterior median estimates should be close to
        self.target = np.array([self.beta_true, self.gamma_true])

        # initial value
        self.theta = np.array([0.4, 0.2])

        # constraints
        EPSILON = np.sqrt(np.finfo(float).eps)

        self.box_bounds = [(EPSILON, 5), (EPSILON, 5)]

        self.y = self.solution[1::, 1:3]

        self.sir_obj = SquareLoss(
            theta=self.theta,
            ode=self.ode,
            x0=self.x0,
            t0=self.t[0],
            t=self.t[1::],
            y=self.y,
            state_name=['I', 'R'])

    def test_single_state_func(self):
        """
        Just to see if the functions manage to run at all
        """
        y = self.solution[1::, 2]
        # test out whether the single state function 'ok'
        sir_obj = SquareLoss(
            self.theta,
            self.ode,
            self.x0,
            self.t[0], self.t[1::],
            y,
            'R')
        
        sir_obj.cost()
        sir_obj.gradient()
        sir_obj.hessian()

    def test_SIR_Estimate_SquareLoss(self):
        y = self.solution[1::, 1:3]
        sir_obj = SquareLoss(
            theta=self.theta,
            ode=self.ode,
            x0=self.x0,
            t0=self.t[0],
            t=self.t[1::],
            y=y,
            state_name=['I', 'R'])

        res_QP = scipy.optimize.minimize(
            fun=sir_obj.cost,
            # jac=sir_obj.sensitivity,
            x0=self.theta,
            # method='SLSQP',
            bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )

    skip("Skipping adjoint test. Maths functions need work.")
    def test_SIR_Estimate_SquareLoss_Adjoint(self):
        y = self.solution[1::, 1:3]

        sir_obj = SquareLoss(
            theta=self.theta,
            ode=self.ode,
            x0=self.x0,
            t0=self.t[0],
            t=self.t[1::],
            y=y,
            state_name=['I', 'R'])

        res_QP = scipy.optimize.minimize(
            fun=sir_obj.cost,
            jac=sir_obj.adjoint,
            x0=self.theta,
            # method='SLSQP',
            bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )

    def test_SIR_Estimate_NormalLoss(self):
        y = self.solution[1::, 1:3]

        sir_obj = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
                             self.t[1::], y, ['I', 'R'])

        res_QP = scipy.optimize.minimize(
            fun=sir_obj.cost,
            # jac=sir_obj.sensitivity,
            x0=self.theta,
            # method='SLSQP',
            bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )

    def tearDown(self):
        self.ode = None
        self.solution = None
        self.x0 = None
        self.t = None
        self.theta = None
        self.target = None


if __name__ == '__main__':
    main()
