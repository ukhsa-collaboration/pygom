from unittest import main, TestCase

from pygom.model.transition import Transition
from pygom.model.deterministic import DeterministicOde

class TestInputSymbols(TestCase):

    def test_Signs(self):
        """
        Making sure that the illegal symbols are catched
        """
        state_list = [['S+'], ['S-'], ['S*'], ['S\\'], ['_S']]
        param_list = ['beta']
        ode = DeterministicOde(['S'], param_list)

        total_fail = 0
        for state in state_list:
            self.assertRaises(AssertionError, DeterministicOde,
                              state, param_list)
        # Happy! :)

    def test_multi_symbol_in_str(self):
        state_list = 'S L I A R D'
        param_list = 'beta,p, kappa,  alpha, f delta  epsilon,N'
        odeList = [
            Transition('S', '- beta * S/N * ( I + delta * A)', 'ODE'),
            Transition('L', 'beta * S/N * (I + delta * A) - kappa * L', 'ODE'),
            Transition(origin='I', equation='p * kappa * L - alpha * I', transition_type='ODE'),
            Transition(origin='A', equation='(1-p) * kappa * L - epsilon * A', transition_type='ODE'),
            Transition(origin='R', equation='f * alpha * I + epsilon * A', transition_type='ODE'),
            Transition(origin='D', equation='(1-f) * alpha * I', transition_type='ODE') 
            ]

        ode = DeterministicOde(state_list, param_list, ode=odeList)
        # this should not throw an error if the model is initialized correctly
        A = ode.get_ode_eqn()


if __name__ == '__main__':
    main()
