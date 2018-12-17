from unittest import main, TestCase

import numpy
import sympy

from pygom import SimulateOde, Transition, TransitionType
from pygom.model import common_models


class TestOdeDecomposition(TestCase):

    def test_simple(self):
        ode1 = Transition('S', '-beta*S*I', 'ode')
        ode2 = Transition('I', 'beta*S*I - gamma * I', 'ode')
        ode3 = Transition('R', 'gamma*I', 'ode')
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        ode = SimulateOde(state_list, param_list, ode=[ode1, ode2, ode3])

        ode2 = ode.get_unrolled_obj()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.get_ode_eqn() - ode2.get_ode_eqn()))

        self.assertTrue(numpy.all(numpy.array(list(diffEqZero))))
#         if numpy.any(numpy.array(list(diffEqZero)) is False):
#             raise Exception("Simple: SIR Decomposition failed")

    def test_hard(self):
        # the SLIARD model is considered to be hard because a state can
        # go to multiple state.  This is not as hard as the SEIHFR model
        # below.
        state_list = ['S', 'L', 'I', 'A', 'R', 'D']
        param_list = ['beta', 'p', 'kappa', 'alpha', 'f', 'delta', 'epsilon', 'N']
        ode_list = [
            Transition('S', '- beta * S/N * ( I + delta * A)', 'ODE'),
            Transition('L', 'beta * S/N * (I + delta * A) - kappa * L', 'ODE'),
            Transition('I', 'p * kappa * L - alpha * I', 'ODE'),
            Transition('A', '(1-p) * kappa * L - epsilon * A', 'ODE'),
            Transition('R', 'f * alpha * I + epsilon * A', 'ODE'),
            Transition('D', '(1-f) * alpha * I', 'ODE')
            ]

        ode = SimulateOde(state_list, param_list, ode=ode_list)

        ode2 = ode.get_unrolled_obj()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.get_ode_eqn() - ode2.get_ode_eqn()))

        self.assertTrue(numpy.all(numpy.array(list(diffEqZero))))

    def test_bd(self):
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'B', 'mu']
        ode_list = [
            Transition(origin='S',
                       equation='-beta * S * I + B - mu * S',
                       transition_type=TransitionType.ODE),
            Transition(origin='I',
                       equation='beta * S * I - gamma * I - mu * I',
                       transition_type=TransitionType.ODE),
            Transition(origin='R',
                       destination='R',
                       equation='gamma * I',
                       transition_type=TransitionType.ODE)
            ]

        ode = SimulateOde(state_list, param_list, ode=ode_list)

        ode2 = ode.get_unrolled_obj()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.get_ode_eqn() - ode2.get_ode_eqn()))

        self.assertTrue(numpy.all(numpy.array(list(diffEqZero))))

    def test_derived_param(self):
        # the derived parameters are treated separately when compared to the
        # normal parameters and the odes
        ode = common_models.Legrand_Ebola_SEIHFR()

        ode_list = [
            Transition('S', '-(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F)'),
            Transition('E', '(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F) - alpha*E'),
            Transition('I', '-gamma_I*(1 - theta_1)*(1 - delta_1)*I - gamma_D*(1 - theta_1)*delta_1*I - gamma_H*theta_1*I + alpha*E'),
            Transition('H', 'gamma_H*theta_1*I - gamma_DH*delta_2*H - gamma_IH*(1 - delta_2)*H'),
            Transition('F', '- gamma_F*F + gamma_DH*delta_2*H + gamma_D*(1 - theta_1)*delta_1*I'),
            Transition('R', 'gamma_I*(1 - theta_1)*(1 - delta_1)*I + gamma_F*F + gamma_IH*(1 - delta_2)*H'),
            Transition('tau', '1')
        ]

        ode1 = SimulateOde(ode.state_list, ode.param_list, ode._derivedParamEqn, ode=ode_list)

        ode2 = ode1.get_unrolled_obj()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.get_ode_eqn() - ode2.get_ode_eqn()))

        self.assertTrue(numpy.all(numpy.array(list(diffEqZero))))


if __name__ == '__main__':
    main()
