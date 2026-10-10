import logging
import sympy

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation, checkEquation
from ..transition import TransitionType

class ODESystem(NumericMethod):
    method_name = 'ode'
    def get_equation(self):
        '''
        Build the algebraic system of ODE's given the transitions and events.
        '''
        # Check if we need to recreate the system
        if self._cache_valid:
            return self._ode
        
        logging.debug('ODE system needs rebuilding')
        # Containers for different contributions to ODE
        between_state_ode = sympy.zeros(self._parent_ode.num_state, 1)
        birth_death_ode = sympy.zeros(self._parent_ode.num_state, 1)
        pure_ode = sympy.zeros(self._parent_ode.num_state, 1)

        # Extract all info from events
        for event in self._parent_ode.event_list:
            rate=checkEquation(event.rate, self._parent_ode)
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, self._parent_ode)
                rate_of_change=magnitude*rate
                if transition.transition_type==TransitionType.B:
                    destination_index=self._parent_ode._state_store.get_index(transition.destination)
                    birth_death_ode[destination_index] += rate_of_change
                elif transition.transition_type==TransitionType.D:
                    origin_index=self._parent_ode._state_store.get_index(transition.origin)
                    birth_death_ode[origin_index] -= rate_of_change
                elif transition.transition_type==TransitionType.T:
                    origin_index=self._parent_ode._state_store.get_index(transition.origin)
                    destination_index=self._parent_ode._state_store.get_index(transition.destination)
                    between_state_ode[origin_index] -= rate_of_change
                    between_state_ode[destination_index] += rate_of_change

        # Now extract any ODE contributions from ODE type transitions
        for ode in self._parent_ode.ode_list:
            origin_index=self._parent_ode._state_store.get_index(ode.origin)
            pure_ode[origin_index] += checkEquation(ode.equation, self._parent_ode)

        # Collect together contributions and make attributes
        self._ode = between_state_ode + birth_death_ode + pure_ode
        self._birthDeathVector = birth_death_ode

        # tests to see whether we have an autonomous system.  Need to
        # convert a non-autonmous system into an autonomous.  Note that
        # we will not do the conversion internally and require the
        # user to do this.  May consider this a feature in the future.
#        # TODO: I think autonomous systems are allowed? Maybe not deterministically?
        for i, eqn in enumerate(self._ode):
#            if self._parent_ode._t in eqn.atoms():      # TODO: maybe this check doesn't work anyway, for namespace reasons?
#                raise Exception("Input is a non-autonomous system. " +
#                                "We can only deal with an autonomous " +
#                                "system at this moment in time")
            self._ode[i], isDifficult = simplifyEquation(eqn)
            # TODO: Do we really need to set this on the parent_ode?
            self._parent_ode._isDifficult = self._parent_ode._isDifficult or isDifficult
            
        return self._ode
