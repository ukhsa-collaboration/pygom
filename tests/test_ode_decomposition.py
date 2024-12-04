from unittest import main, TestCase

import numpy
import sympy

from pygom import SimulateOde, Transition, TransitionType
from pygom.model import common_models


class TestOdeDecomposition(TestCase):
    def test_simple(self):
        ode1 = Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.ODE)
        ode2 = Transition(origin='I', equation='beta*S*I - gamma * I', transition_type=TransitionType.ODE)
        ode3 = Transition(origin='R', equation='gamma*I', transition_type=TransitionType.ODE)
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
            Transition(origin='S', equation='- beta * S/N * ( I + delta * A)', transition_type=TransitionType.ODE),
            Transition(origin='L', equation='beta * S/N * (I + delta * A) - kappa * L', transition_type=TransitionType.ODE),
            Transition(origin='I', equation='p * kappa * L - alpha * I', transition_type=TransitionType.ODE),
            Transition(origin='A', equation='(1-p) * kappa * L - epsilon * A', transition_type=TransitionType.ODE),
            Transition(origin='R', equation='f * alpha * I + epsilon * A', transition_type=TransitionType.ODE),
            Transition(origin='D', equation='(1-f) * alpha * I', transition_type=TransitionType.ODE)
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
            Transition(origin='S', equation='-(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F)/N'),
            Transition(origin='E', equation= '(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F)/N - alpha*E'),
            Transition(origin='I', equation= '-gamma_I*(1 - theta_1)*(1 - delta_1)*I - gamma_D*(1 - theta_1)*delta_1*I - gamma_H*theta_1*I + alpha*E'),
            Transition(origin='H', equation= 'gamma_H*theta_1*I - gamma_DH*delta_2*H - gamma_IH*(1 - delta_2)*H'),
            Transition(origin='F', equation= '- gamma_F*F + gamma_DH*delta_2*H + gamma_D*(1 - theta_1)*delta_1*I'),
            Transition(origin='R', equation= 'gamma_I*(1 - theta_1)*(1 - delta_1)*I + gamma_F*F + gamma_IH*(1 - delta_2)*H')
        ]

        ode1 = SimulateOde(ode.state_list, ode.param_list, ode._derivedParamEqn, ode=ode_list)

        ode2 = ode1.get_unrolled_obj()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.get_ode_eqn() - ode2.get_ode_eqn()))

        self.assertTrue(numpy.all(numpy.array(list(diffEqZero))))


if __name__ == '__main__':
    main()
