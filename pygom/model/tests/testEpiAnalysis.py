from unittest import TestCase

from pygom import common_models, epi_analysis
import sympy

class TestEpiAnalysis(TestCase):

    def test_simple(self):
        '''
        This actually only test the internal consistency of the functions
        rather than the actual correctness of the result.  Nevertheless,
        it is a valid test and we will use this hack for now because
        testing the result of a sympy object against strings is now the
        simplest procedure.
        '''
        ode = common_models.SIR_Birth_Death()
        diseaseState = ['I']
        R0 = epi_analysis.getR0(ode, ['I'])
        
        F, V = epi_analysis.getDiseaseProgressionMatrices(ode, diseaseState)
        e = epi_analysis.getR0GivenMatrix(F, V)
        dfe = epi_analysis.getDFE(ode, ['I'])
        if (sympy.simplify(R0 - e[0].subs(dfe)) != 0):
            raise Exception("Simple: Epi Analysis failed")
        