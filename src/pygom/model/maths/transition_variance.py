import sympy

from .mathsmethod import NumericMethod

class TransitionVariance(NumericMethod):
    method_name = 'transition_variance'
    def get_equation(self):
        '''
        This is the variance of the changes in the transition rates
        (aka propensity funtions) after a potential timestep:
        equations (8b) from 
        https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        For n transitions there is a vectors of length n.
        '''

        F = self._parent_ode.transition_jacobian.get_equation()

        sigma2 = sympy.zeros(self._parent_ode.num_events, 1)
        for event_index_i in range(self._parent_ode.num_events):
            for event_index_j, rate_j in enumerate(self._parent_ode.event_rate_vector.get_equation()):
                sigma2[event_index_i] += F[event_index_i, event_index_j] * F[event_index_i, event_index_j] * rate_j

        return sigma2


class TransitionVarianceMatrix(NumericMethod):
    method_name = 'transition_variance_matrix'
    def get_equation(self):
        '''
        This is the variance of the changes in the transition rates
        (aka propensity funtions) after a potential timestep:
        equations (8b) from 
        https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        For n transitions there is a vectors of length n.
        '''

        F = self._parent_ode.transition_jacobian.get_equation()
        rates = self._parent_ode.event_rate_vector.get_equation()

        timestep = sympy.zeros(self._parent_ode.num_events, self._parent_ode.num_events)
        for event_index_i in range(self._parent_ode.num_events):
            for event_index_j in range(self._parent_ode.num_events):
                timestep[event_index_i, event_index_j] = (rates[event_index_i]**2 / (rates[event_index_j] * F[event_index_i, event_index_j]**2) )

        return timestep
    
