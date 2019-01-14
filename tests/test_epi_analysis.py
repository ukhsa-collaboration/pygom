from unittest import main, TestCase

import sympy

from pygom.model import common_models, epi_analysis 


class TestEpiAnalysis(TestCase):

    def test_simple(self):
        """
        This actually only test the internal consistency of the functions
        rather than the actual correctness of the result.  Nevertheless,
        it is a valid test and we will use this hack for now because
        testing the result of a sympy object against strings is now the
        simplest procedure.
        """
        ode = common_models.SIR_Birth_Death()
        disease_state = ['I']
        R0 = epi_analysis.R0(ode, ['I'])

        F, V = epi_analysis.disease_progression_matrices(ode, disease_state)
        e = epi_analysis.R0_from_matrix(F, V)
        dfe = epi_analysis.DFE(ode, ['I'])
        self.assertTrue(sympy.simplify(R0 - e[0].subs(dfe)) == 0)


if __name__ == '__main__':
    main()
