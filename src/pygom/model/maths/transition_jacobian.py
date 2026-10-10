import sympy

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class TransitionJacobian(NumericMethod):
    method_name = 'transition_jacobian'
    def get_equation(self):
        '''
        Evaluate equation (7) from 
        https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        where F_[i,j] is the change in transition rate a[i], if a transition of 
        type j occurs:
        F_[i,j] = sum_k diff(a[i], x_k) v_[k,j]
        where k=state and v[k,j] is how much state x_k changes by if transition
        of type j occurs.
        '''

        F = sympy.zeros(self._parent_ode.num_events, 
                        self._parent_ode.num_events)

        for event_index_i, rate in enumerate(self._parent_ode.event_rate_vector.get_equation()):
            for event_index_j in range(self._parent_ode.num_events):
                for state_index, state in enumerate(self._parent_ode._iterStateList()):             
                    diffEqn, isDifficult = simplifyEquation( sympy.diff(rate, state, 1)  )  # diff(a_i, x_k)
                    F[event_index_i, event_index_j] += diffEqn*self._parent_ode.state_change_matrix.get_equation()[state_index, event_index_j]
                    self._parent_ode._isDifficult = self._parent_ode._isDifficult or isDifficult

        return F