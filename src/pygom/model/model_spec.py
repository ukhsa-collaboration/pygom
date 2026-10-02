import sympy
import numpy as np

from .transition import Event, Transition, TransitionType
from ._model_errors import InputError, OutputError
from ._model_verification import checkEquation
from .ode_variable import ODEVariable
from . import ode_utils

### Main Classes ###
class HasNewTransition(ode_utils.CompileCanary):
    states = []


# TODO: Is there a point at which we trust that all the parameter orders are agreed on?
# We can proceed to pass state/param information to solving/fitting methods as lists
# safe in the knowledge that we can unpack them on the other side. I guess this belongs
# in the stores and we just need to ensure that caching keeps eveything up to date


# TODO: Think about which module is responsible for the unpacking of certain features.
# For example, if I want a list of parameter values from the store, do I build this
# outside of the store or do I add a @property within the store to present it.


"""
BaseOdeModel becomes modelspec?

This gethers states, params, events and 

"""

class ModelSpec(object):
    """
    This base object stores the defining objects of a compartmental model
    and has functions to verify the build

    Parameters
    ----------
    state: list
        A list of states (string)
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string, string))
    transition: list
        A list of transition (:class:`.Transition`)
    event: list
        A list of events (:class:`.Transition`)
    birth_death: list
        A list of birth or death process (:class:`.Transition`)
    ode: list
        A list of ode (:class:`.Transition`)

    """

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            event=None,
        ):

        self._sp = None
        self._parameter_store = None
        self._state_store = None
        self._derived_parameter_store = None

        # we always need time to be a symbol and it should be denoted as t
        # TODO: should be added to state / param store?
        self._t = sympy.symbols('t', real=True)

        ## Parameters ##
        self.set_parameters(param)

        ## States ##
        self.set_states(state)

        ## Derived Parameters ##
        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        self.set_derived_parameters(derived_param)

        ## Transitions ##
        self.set_events(event)

    # def __repr__(self):
    #     return f'{self.__class__.__name__ } {self._get_model_str()}'

    def _invalidate_caches(self)->None:
        """
        Tell objects that have cached components to reset their caches as
        the underlying system has changed

        TODO: this needs to be communicated back to the maths objects
        """
        
        self._sp = None

    ###########################################################################
    #
    # States and params
    #
    ###########################################################################

    ## Building and modifying sotres ##

    ###################################################################################
    ## Store initialisers

    def set_states(self, state_list:list[str|ODEVariable])->None:
        """
        Declare the states for the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """

        if state_list is None:
            state_list = []

        # create a new store to replace the existing (if creations succeds)
        new_state_store = ode_utils.StateStore()
        new_state_store.add(state_list, self.states_and_parameters_dict)
        
        self._state_store = new_state_store
        self._invalidate_caches()

    def set_parameters(self, parameter_list:list[str|ODEVariable]) -> None:
        """
        Declare and store the parameter names for the compartmental model

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """

        if parameter_list is None:
            parameter_list = []

        # create a new store to replace the existing (if creations success)
        new_parameter_store = ode_utils.ParameterStore()
        new_parameter_store.add(parameter_list, self.states_and_parameters_dict)
        
        self._parameter_store = new_parameter_store
        self._invalidate_caches()

    def set_derived_parameters(self, derived_parameter_list:list[tuple[str|ODEVariable]])->None:
        """
        Declare the derived parameters

        """

        if derived_parameter_list is None:
            derived_parameter_list = []

        # create a new store to replace the existing (if creations succeds)
        new_derived_parameter_store = ode_utils.DerivedParameterStore()

        new_derived_parameter_store.add(derived_parameter_list, self.states_and_parameters_dict)
        self._derived_parameter_store = new_derived_parameter_store
        self._invalidate_caches()

    ###################################################################################
    ## Modify stores

    def append_parameters(self, parameter_list:list[str|ODEVariable])->None:
        """
        Append additional parameters to the ode system

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter to be 
            added
        """
        # create a new store (if we don't already have one)
        if self._parameter_store is None:
            new_parameter_store = ode_utils.ParameterStore()
        else:
            new_parameter_store = self._parameter_store
        new_parameter_store._add_to_store(parameter_list)
        
        self._parameter_store = new_parameter_store
        self._invalidate_caches()

    ## Accessing and setting properties ##

    ###################################################################################
    ## Values
    # TODO: check which of these are O(1) and O(N) and maybe add to doctring

    @property
    def state(self):
        """
        Returns
        -------
        list
            state in symbol with current value,
            (:mod:`sympy.core.symbol`,numeric)

        """
        return [(symb, val) for symb, val in zip(self._state_store.symbol_list,
                                                 self._state_store.values)]
    
    @state.setter
    def state(self,
              states:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        """

        """
        self._state_store.values = states

    @property
    def parameters(self):
        """
        Returns
        -------
        list
            A list which contains tuple of two elements, parameter symbol and its value
            (:mod:`sympy.core.symbol`, numeric)

        """
        if self._parameter_store.all_values_set:
            return [(symb, val) for symb, val in zip(self._parameter_store.symbol_list,
                                                     self._parameter_store.values)]

    @parameters.setter
    def parameters(self, 
                   parameters:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        """
        Set the values for the parameters already defined.  Note that unless
        the parameters are entered via a dictionary or a two element list,tuple
        we assume that it is in the order of :meth:`.getParamList`

        Parameters
        ----------
        parameters: dict of {parameter_ID: parameter_value} (prefered) _or_
            a list which contains elements made of 2 element tuples 
            (string, numeric value) _or_ a single array like object with
            length equal to the number of parameters, in the same order as they
            were created.
        """
        self._parameter_store.values = parameters

    ###################################################################################
    ## Symbols

    @property
    def state_list(self):
        """
        Returns a list of the states in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._state_store.symbol_list

    @property
    def param_list(self):
        """
        Returns a list of the parameters in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._parameter_store.symbol_list

    @property
    def derived_param_list(self):
        """
        Returns a list of the derived parameters in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._derived_parameter_store.symbol_list

    ###################################################################################
    ## Store counts

    @property
    def num_state(self):
        """
        Returns the number of state

        Returns
        -------
        int
            the number of states

        """
        return len(self._state_store)

    @property
    def num_param(self):
        """
        Returns the number of parameters

        Returns
        -------
        int
            the number of parameters

        """
        return len(self._parameter_store)

    @property
    def num_derived_param(self):
        """
        Returns the number of derived parameters

        Returns
        -------
        int
            the number of derived parameters

        """
        return len(self._derived_parameter_store)

    ###################################################################################
    ## Namespace

    def _generate_state_dict(self)->None:
        '''
        Create state dict
        '''
        states = {}
        if self._state_store:
            states = self._state_store.symbol_dict
        self._state_dict = states

    @property
    def state_dict(self)->dict[str: sympy.Symbol]:
        '''
        State dict
        '''
        if self._sp is None:
            self._generate_state_dict()
        return self._state_dict

    def _generate_parameter_dict(self)->None:
        '''
        Create parameter dict
        '''
        parameters = {}
        if self._parameter_store:
            parameters = self._parameter_store.symbol_dict
        self._parameter_dict = parameters

    @property
    def parameter_dict(self)->dict[str: sympy.Symbol]:
        '''
        Parameter dict
        '''
        if self._sp is None:
            self._generate_parameter_dict()
        return self._parameter_dict

    def _generate_derived_parameter_dict(self)->None:
        '''
        Create derived parameter dict
        '''
        derived_parameters = {}
        if self._derived_parameter_store:
            derived_parameters = self._derived_parameter_store.symbol_dict
        self._derived_parameter_dict = derived_parameters

    @property
    def derived_parameter_dict(self)->dict[str: sympy.Symbol]:
        '''
        Derived parameter dict
        '''
        if self._sp is None:
            self._generate_derived_parameter_dict()
        return self._derived_parameter_dict

    def _generate_derived_parameter_expression_dict(self)->None:
        '''
        Create derived parameter dict, from id to expression
        '''
        derived_parameter_expressions = {}
        if self._derived_parameter_store:
            derived_parameter_expressions = self._derived_parameter_store.expression_dict
        self._derived_parameter_expression_dict = derived_parameter_expressions

    @property
    def derived_parameter_expression_dict(self)->dict[str: sympy.Symbol]:
        '''
        Derived parameter dict
        '''
        if self._sp is None:
            self._generate_derived_parameter_expression_dict()
        return self._derived_parameter_expression_dict

    def _generate_states_and_parameters(self)->None:
        '''
        Creates the entire collection of symbols
        '''

        self._sp = (
            self.state_dict |
            self.parameter_dict |
            self.derived_parameter_dict |
            {'t': self._t}
        )

    @property
    def states_and_parameters_dict(self)->dict[str: sympy.Symbol]:
        '''
        An attribute collecting together all the states and variables in sympy
        form. This is used for the check equation function.
        '''
        if self._sp is None:
            self._generate_states_and_parameters()

        return self._sp

    @property
    def states_and_parameters_list(self)->list[sympy.Symbol]:
        '''
        An attribute collecting together all the states and variables in sympy
        form. This is used for the autowrap method
        TODO: this is not only states and parameters but derived parameters too, maybe rename to total namespace
        '''
        return list(self.states_and_parameters_dict.values())

    ###################################################################################
    ## Index look up

    def get_state_index(self, input_str:str)->int:
        """
        Finds the index of the state

        Returns
        -------
        int
            the index of the desired state

        """
        if isinstance(input_str, str):
            return self._state_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._state_store.get_index(x) for x in input_str]

    def get_parameter_index(self, input_str:str)->int:
        """
        Finds the index of the parameter

        Returns
        -------
        int
            the index of the desired parameter
        """
        if isinstance(input_str, str):
            return self._parameter_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._parameter_store.get_index(x) for x in input_str]

    def get_derived_parameter_index(self, input_str:str)->int:
        """
        Finds the index of the derived parameter

        Returns
        -------
        int
            the index of the desired derived parameter
        """
        if isinstance(input_str, str):
            return self._derived_parameter_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._derived_parameter_store.get_index(x) for x in input_str]

    ###################################################################################
    ## State limits

    @property
    def state_lower_limits(self):
        """
        
        """
        return self._state_store.lower_limit_list

    @property
    def state_upper_limits(self):
        """
        
        """
        return self._state_store.upper_limit_list

    ##########################################################
    #
    # Transitions and Events
    #
    ##########################################################

    # TODO: Transition store depends on some featurs of params,
    # does this hierarchy change anything?

    # TODO: Event indexing might be necessary, can we guarantee everything is in order always?

    ###################################################################################
    ## Store initialiser

    def set_events(self, event_list:list[Transition|Event]|None)->None:
        """
        Declare the transitions for the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """
        if event_list is None:
            event_list = []

        # create a new store to replace the existing (if creations succeds)
        new_event_store = ode_utils.EventStore()
        new_event_store.add(event_list, self.states_and_parameters_dict, self.state_dict)
        
        self._event_store = new_event_store
        self._invalidate_caches()

    @property
    def event_list(self):
        """
        Returns a list of the events

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        return self._event_store.event_list

    ###################################################################################
    ## Store counts

    @property
    def num_event(self):
        """
        Returns the number of events

        Returns
        -------
        int
            the number of events

        """
        return self._event_store.num_events

    # TODO: This doesn't actually contain any extra info than the reaction matrix
    #       consider what to do with it.
    # def get_ReactantMatrix(self):
    #     """
    #     The reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is involved in transition j, \\\\
    #                                    0, &otherwise \\right.
    #     """
    #     # declare holder
    #     self._lambdaMat = np.zeros((self.num_state, self.num_events), int)

    #     for event_index, event in enumerate(self.event_list):
    #         for transition in event.transition_list:
    #             if transition.transition_type==TransitionType.B:
    #                 destination_index = self.get_state_index(transition.destination)
    #                 self._lambdaMat[destination_index, event_index] = 1
    #             elif transition.transition_type==TransitionType.D:
    #                 origin_index = self.get_state_index(transition.origin)
    #                 self._lambdaMat[origin_index, event_index] = 1
    #             elif transition.transition_type==TransitionType.T:
    #                 origin_index = self.get_state_index(transition.origin)
    #                 destination_index = self.get_state_index(transition.destination)
    #                 self._lambdaMat[origin_index, event_index] = 1
    #                 self._lambdaMat[destination_index, event_index] = 1

    #     return self._lambdaMat
