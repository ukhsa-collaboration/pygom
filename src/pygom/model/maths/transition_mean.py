import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class TransitionMean(NumericMethod):
    method_name = 'transition_mean'
    def get_equation(self):
        '''
        This is the mean of the changes in the transition rates
        (aka propensity funtion) after a potential timestep:
        equations (8a) from 
        https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        For n transitions there is a vectors of length n.
        '''
        F = self._parent_ode.transition_jacobian.get_equation()

        mu = sympy.zeros(self._parent_ode.num_events, 1)
        for event_index_i in range(self._parent_ode.num_events):
            for event_index_j, rate_j in enumerate(self._parent_ode.event_rate_vector.get_equation()):
                mu[event_index_i] += F[event_index_i, event_index_j] * rate_j

        # TODO: Propensity functions also change if there is time dependence
        #       This will be addressed better in the next version where tau
        #       leaping will be updated.
        # # If time dependence, add in another term to reflect this:
        # timelike_symbols=[symb for symb in eqn_i.free_symbols if str(symb)=='t']
        # is_time_dependent=len(timelike_symbols)>0
        # if is_time_dependent and self.tstep:
        #     time_variable = [timelike_symbols][0]
        #     mu[i] += sympy.diff(eqn_i, time_variable, 1) # mean changes but sd does not, TODO: check this

        return  mu