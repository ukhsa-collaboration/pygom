from unittest import main, TestCase

import copy

import numpy as np
import scipy.integrate

from pygom.model import common_models


class TestJacobians(TestCase):

    def setUp(self):
        self.h = np.sqrt(np.finfo(np.float).eps)
        # initial time
        self.t0 = 0
        # the initial state, normalized to zero one
        self.x0 = np.array([1, 1.27e-6, 0])
        # params
        param_eval = [('beta', 0.5), ('gamma', 1.0 / 3.0)]
        self.ode = common_models.SIR(param_eval)
        self.ode.initial_values = (self.x0, self.t0)
        self.d = self.ode.num_state
        self.p = self.ode.num_param
        self.t = np.linspace(0, 150, 100)
        self.index = np.random.randint(100)

    def tearDown(self):
        self.ode = None
        self.t = None

    def test_odeJacobian(self):
        """
        Analytic Jacobian for the ode against the forward
        differencing numeric Jacobian
        """
        # integrate without using the analytical Jacobian
        solution, _output = scipy.integrate.odeint(self.ode.ode,
                                                   self.x0, self.t,
                                                   full_output=True)

        # the Jacobian of the ode itself
        ff0 = solution[self.index, :]
        J0 = self.ode.ode(ff0, self.t[self.index])
        J = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                ff_temp = copy.deepcopy(ff0)
                ff_temp[j] += self.h
                J[i,j] = (self.ode.ode(ff_temp, self.t[self.index])[i] - J0[i])/self.h

        JAnalytic = self.ode.jacobian(ff0, self.t[self.index])
        self.assertTrue(np.allclose(J, JAnalytic))

    def test_SensJacobian(self):
        """
        Analytic Jacobian for the forward sensitivity equations against
        the forward differencing numeric Jacobian
        """
        # initial conditions
        s0 = np.zeros(self.d*self.p)
        ffParam = np.append(self.x0, s0)

        # integrate without using the analytical Jacobian
        solution_sens, _out = scipy.integrate.odeint(self.ode.ode_and_sensitivity,
                                                     ffParam, self.t,
                                                     full_output=True)

        # the Jacobian of the ode itself
        ff0 = solution_sens[self.index, :]
        J0 = self.ode.ode_and_sensitivity(ff0, self.t[self.index])
        p1 = self.p + 1
        dp1 = self.d*p1
        J = np.zeros((dp1, dp1))
        for i in range(dp1):
            for j in range(dp1):
                ff_temp = copy.deepcopy(ff0)
                ff_temp[j] += self.h
                J[i,j] = (self.ode.ode_and_sensitivity(ff_temp, self.t[self.index])[i] - J0[i])/self.h

        JAnalytic = self.ode.ode_and_sensitivity_jacobian(ff0, self.t[self.index])
        # JAnalytic = ode.odeAndSensitivityJacobian(ff0,t[index])

        self.assertTrue(np.allclose(J, JAnalytic))

    def test_HessianJacobian(self):
        """
        Analytic Jacobian for the forward forward sensitivity equations
        i.e. the Hessian of the objective function against
        the forward differencing numeric Jacobian
        """
        ff0 = np.zeros(self.d*self.p*self.p)
        s0 = np.zeros(self.d*self.p)
        ffParam = np.append(np.append(self.x0, s0), ff0)
        # our integration
        sol_hess, _o = scipy.integrate.odeint(self.ode.ode_and_forwardforward,
                                              ffParam, self.t, full_output=True)

        numFF = len(ffParam)
        J = np.zeros((numFF, numFF))
        # get the info
        ff0 = sol_hess[self.index, :]
        # evaluate at target point
        J0 = self.ode.ode_and_forwardforward(ff0, self.t[self.index])
        # J0 = ode.odeAndForwardforward(ff0, t[index])
        # the Analytical solution is
        JAnalytic = self.ode.ode_and_forwardforward_jacobian(ff0, self.t[self.index])
        # JAnalytic = ode.odeAndForwardforwardJacobian(ff0, t[index])
        # now we go and find the finite difference Jacobian
        for i in range(numFF):
            for j in range(numFF):
                ff_temp = copy.deepcopy(ff0)
                ff_temp[j] += self.h
                J[i,j] = (self.ode.ode_and_forwardforward(ff_temp, self.t[self.index])[i] - J0[i])/self.h


if __name__ == '__main__':
    main()
