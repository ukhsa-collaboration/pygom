from types import NoneType
from collections import OrderedDict
from indexed import IndexedOrderedDict

import numpy as np

from sympy import Symbol, symbols

from pygom.model._model_errors import InputError

from pygom.model.transition import Transition, Event

from .._model_verification import checkEquation

__all__ = ['EventStore']

class EventStore(object):
    '''

    Event registry

    An object to hold events for a PyGOM compartmental model system

    This object is designed to:
    * Store all the events
    * That's it? State change matrix takes care of everything else?
    * Manage transitions (for now, no management necessary unless we want
        to forbid e.g. duplicates -however, might be the case that 2 independent
        transitions can exist with same origin, destination and rate) 
    '''

    def __init__(
            self,
            events=None
        ):
        '''
        The init method
        '''
        # self._events = IndexedOrderedDict()

        self._accepted_variable_types = (Transition, Event)

        self._event_list = list()

        if events is not None:
            self.add(events)

    #####################################
    # User building / modifications
    #####################################

    def _force_list(self, variable):
        """
        Convert var into [var] and leave [x, y, z] as it is
        """
        if isinstance(variable, self._accepted_variable_types):
            return [variable]
        if isinstance(variable, list):
            return variable
        raise InputError(
            f"Variable must be list or {self._accepted_variable_types}."
        )

    @property
    def num_events(self)->int:
        '''
        The current number of variables in the store
        '''
        return len(self._event_list)

    def add(
            self, 
            events:Event|list[Event],
            variable_namespace,
            state_namespace
        ):
        '''
        Add variable(s) to the store

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
            of the list of variables
        '''

        events = self._force_list(events)

        for event in events:
            if isinstance(event, Transition):
                event = Event(
                    rate=event.rate,
                    transition_list=[event]
                )

            event.rate_expression = checkEquation(event.rate, variable_namespace)

            for transition in event.transition_list:
                if transition.origin:
                    if not transition.origin in state_namespace:
                        raise(InputError(f"Origin '{transition.origin}' has not been declared as a state"))
                if transition.destination:
                    if not transition.destination in state_namespace:
                        raise(InputError(f"Destination '{transition.destination}' has not been declared as a state"))

                transition.magnitude_expression = checkEquation(transition.magnitude, variable_namespace)

            self._event_list.append(event)

    # def remove()

    #####################################
    # Externally shared properties
    #####################################

    @property
    def event_list(self)->list[str]:
        '''
        Get a list of string variable names in the order they were added
        '''
        return self._event_list
