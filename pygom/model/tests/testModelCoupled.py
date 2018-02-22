from unittest import TestCase

from collections import OrderedDict

from pygom import DeterministicOde, Transition, TransitionType
import numpy
import sympy
# import six

## define parameters
param_eval = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
             'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'p':0.01}

class TestModelCoupled(TestCase):

    def test_compareAll(self):
        '''
        Compare the solution of a coupled ode using different ways of defining it
        '''

        n = 2

        solution1 = self.naive(n)
        solution2 = self.shorter(n)
        solution3 = self.even_shorter(n)
        solution4 = self.very_short(n)
        solution5 = self.confused(n)

        self.assertTrue(numpy.allclose(solution1, solution2))
        # if numpy.any((solution1 - solution2) >= 0.001):
        #     raise Exception("Solution not match")

        self.assertTrue(numpy.allclose(solution2, solution3))
        # if numpy.any((solution3 - solution2) >= 0.001):
        #     raise Exception("Solution not match")

        self.assertTrue(numpy.allclose(solution3, solution4))
        # if numpy.any((solution4 - solution3) >= 0.001):
        #     raise Exception("Solution not match")

        self.assertTrue(numpy.allclose(solution4, solution5))
        # if numpy.any((solution5 - solution4) >= 0.001):
        #     raise Exception("Solution not match")

    def naive(self, n):
        # n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []
        N, S, E, I, R = [], [], [], [], []

        for i in s:
            N += ['N_'+i]
            S += ['S_'+i]
            E += ['E_'+i]
            I += ['I_'+i]
            R += ['R_'+i]
            lambdaTemp = '0 '
            for j in s:
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'* beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+' * beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        stateList = S + E + I + R

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i], lambdaStr[i])]
            transitionList += [Transition(origin=S[i], destination=E[i], equation=lambdaName[i]+ '*' +S[i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=E[i], destination=I[i], equation=' epsilon * ' +E[i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=I[i], destination=R[i], equation=' gamma * ' +I[i] ,transition_type=TransitionType.T)]
            bdList += [Transition(origin=S[i], equation='d * '+S[i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=E[i], equation='d * '+E[i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=I[i], equation='d * '+I[i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=R[i], equation='d * '+R[i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=S[i], equation='d * '+N[i], transition_type=TransitionType.B)]

        ode = DeterministicOde(stateList,
                               paramList,
                               derived_param=derivedParamList,
                               transition=transitionList,
                               birth_death=bdList)

        t = numpy.linspace(0, 40, 100)
        x01 = self.getInitialValue(paramList, n)

        ode.parameters = param_eval
        ode.initial_values = (numpy.array(x01,float),t[0])
        solution1 = ode.integrate(t[1::])
        return solution1

    def shorter(self, n):
        # n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []

        stateName = ["S", "E", "I", "R"]
        states = OrderedDict.fromkeys(stateName,[])
        N =  []

        for i in s:
            for v in states:
                states[v] = states[v]+[str(v)+"_"+i]
            N += ['N_'+i]
            lambdaTemp = '0'
            for j in s:
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        stateList = []
        for v in states: stateList += states[v]

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origin=states['S'][i], destination=states['E'][i], equation=lambdaName[i]+ '*' +states['S'][i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=states['E'][i], destination=states['I'][i], equation=' epsilon * ' +states['E'][i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=states['I'][i], destination=states['R'][i], equation=' gamma * ' +states['I'][i] ,transition_type=TransitionType.T)]
            bdList += [Transition(origin=states['S'][i], equation='d * '+states['S'][i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=states['E'][i], equation='d * '+states['E'][i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=states['I'][i], equation='d * '+states['I'][i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=states['R'][i], equation='d * '+states['R'][i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=states['S'][i], equation='d * '+N[i], transition_type=TransitionType.B)]

        ode = DeterministicOde(stateList,
                               paramList,
                               derived_param=derivedParamList,
                               transition=transitionList,
                               birth_death=bdList)

        t = numpy.linspace(0, 40, 100)
        x01 = self.getInitialValue(paramList, n)

        ode.parameters = param_eval
        ode.initial_values = (numpy.array(x01,float),t[0])
        solution2 = ode.integrate(t[1::])

        return solution2

    def even_shorter(self, n):
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []

        stateName = ["S", "E", "I", "R"]
        states = OrderedDict.fromkeys(stateName,[])
        N =  []

        for i in s:
            for v in states:
                states[v] = states[v]+[str(v)+"_"+i]
            N += ['N_'+i]
            lambdaTemp = '0'
            for j in s:
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        stateList = []
        for v in states: stateList += states[v]

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origin=states['S'][i], destination=states['E'][i], equation=lambdaName[i]+ '*' +states['S'][i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=states['E'][i], destination=states['I'][i], equation=' epsilon * ' +states['E'][i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=states['I'][i], destination=states['R'][i], equation=' gamma * ' +states['I'][i] ,transition_type=TransitionType.T)]
            for v in states:
                bdList += [Transition(origin=states[v][i], equation='d * '+states[v][i], transition_type=TransitionType.D)]
            bdList += [Transition(origin=states['S'][i], equation='d * '+N[i], transition_type=TransitionType.B)]

        ode = DeterministicOde(stateList,
                               paramList,
                               derived_param=derivedParamList,
                               transition=transitionList,
                               birth_death=bdList)

        t = numpy.linspace(0, 40, 100)
        x01 = self.getInitialValue(paramList, n)

        ode.parameters = param_eval
        ode.initial_values = (numpy.array(x01,float),t[0])
        solution3 = ode.integrate(t[1::])

        return solution3

    def very_short(self, n):
        beta = []
        lambdaStr = []
        lambdaName = []

        var_dict = globals()
        stateName = ['N', 'S', 'E', 'I', 'R']
        for s in stateName:
            # six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
            # glb[s] = [s+'_'+str(i) for i in range(n)]
            var_dict[s] = [s+'_'+str(i) for i in range(n)]
        # print(glb.keys())
        # print(lcl.keys())

        for i in range(n):
            lambdaTemp = '0 '
            for j in range(n):
                beta.append('beta_%s%s' % (i,j))
                lambdaTemp += '+ I_%s * beta_%s%s ' % (j, i, j)
                if i != j:
                    lambdaTemp += ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+str(i)]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        stateList = S + E + I + R

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origin=S[i], destination=E[i], equation=lambdaName[i]+ '*' +S[i], transition_type=TransitionType.T)]
            transitionList += [Transition(origin=E[i], destination=I[i], equation=' epsilon * ' +E[i], transition_type=TransitionType.T)]
            transitionList += [Transition(origin=I[i], destination=R[i], equation=' gamma * ' +I[i], transition_type=TransitionType.T)]

            bdList += [Transition(origin=S[i], equation='d * '+N[i], transition_type=TransitionType.B)]
        for s in stateList:
            bdList += [Transition(origin=s, equation='d * '+s, transition_type=TransitionType.D)]

        ode = DeterministicOde(stateList,
                               paramList,
                               derived_param=derivedParamList,
                               transition=transitionList,
                               birth_death=bdList)

        t = numpy.linspace(0, 40, 100)
        x01 = self.getInitialValue(paramList, n)

        ode.parameters = param_eval
        ode.initial_values = (numpy.array(x01,float),t[0])
        solution4 = ode.integrate(t[1::])

        return solution4

    def confused(self, n):
        # stateName = ['N', 'S', 'E', 'I', 'R']
#         for s in stateName:
#             six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
        var_dict = globals()
        stateName = ['N', 'S', 'E', 'I', 'R']
        for s in stateName:
            # six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
            # glb[s] = [s+'_'+str(i) for i in range(n)]
            var_dict[s] = [s+'_'+str(i) for i in range(n)]


        beta = []
        bdList = list()
        transitionList = list()
        derivedParamList = list()
        for i in range(n):
            lambdaStr = '0 '
            for j in range(n):
                beta.append('beta_%s%s' % (i,j))
                lambdaStr += '+ I_%s * beta_%s%s ' % (j, i, j)
                if i != j:
                    lambdaStr += ' * p'
            derivedParamList += [('lambda_'+str(i), lambdaStr)]

            transitionList += [Transition(origin=S[i], destination=E[i],equation='lambda_'+str(i)+ '*' +S[i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=E[i], destination=I[i],equation=' epsilon * ' +E[i] ,transition_type=TransitionType.T)]
            transitionList += [Transition(origin=I[i], destination=R[i],equation=' gamma * ' +I[i] ,transition_type=TransitionType.T)]
            bdList += [Transition(origin=S[i], equation='d * '+N[i], transition_type=TransitionType.B)]

        stateList = S + E + I + R
        for s in stateList:
            bdList += [Transition(origin=s, equation='d * '+s, transition_type=TransitionType.D)]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        ode = DeterministicOde(stateList,
                               paramList,
                               derived_param=derivedParamList,
                               transition=transitionList,
                               birth_death=bdList)

        t = numpy.linspace(0, 40, 100)
        x01 = self.getInitialValue(paramList, n)

        ode.parameters = param_eval
        ode.initial_values = (numpy.array(x01,float),t[0])
        solution5 = ode.integrate(t[1::])

        return solution5

    def getInitialValue(self, param_list, n=2):
        '''
        Finds the initial values where the stationary condition is achieved
        '''
        var_dict = globals()
        ## to find the stationary starting conditions
        for param in param_list:
            var_dict[param] = sympy.symbols(param)

        N = sympy.symbols("N")

        R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)
        S = N/R0
        E = (d*N) / (d + epsilon) * (1 - 1/R0)
        I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)
        R = N - S - E - I

        paramEval1 = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
                      'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'N':10**6}

        x0 = [S.subs(paramEval1),E.subs(paramEval1),I.subs(paramEval1),R.subs(paramEval1)]
        x01 = []
        for s in x0:
            x01 += [s] * n

        return x01
