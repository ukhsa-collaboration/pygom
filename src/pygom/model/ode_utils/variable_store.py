from types import NoneType
from collections import OrderedDict
from indexed import IndexedOrderedDict

import numpy as np

from sympy import Symbol, symbols

from pygom.model.ode_variable import ODEVariable, State, Parameter, DerivedParameter, CallableParameter
from pygom.model._model_errors import InputError

from scipy.stats._distn_infrastructure import rv_frozen

from numbers import Number

from abc import abstractmethod

__all__ = [
    'VariableStore',
    'ParameterStore',
    'StateStore',
    'DerivedParameterStore'
]

"""
Inheritance hierarchy may still be useful. 
Forcing states and params to expose identical value-management behaviour
might be making design awkward
"""


# TODO: it is quite annoying to sometimes call things by id (string) and
# sometimes by symbol, or to want things as id/symbol. Might have to just pick one way?

# TODO: cache for speed, not necessarily to just keep everything up to date


class IndexShim(object):
    """
    TODO: docstring
    """
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, item:int):
        return self.parent._variables.values()[item]

class VariableStore(object):
    '''
    Variable (state and parameter) registry.
    Parent class for parameter and state stores.
    An object to hold variables for a PyGOM compartmental model system

    This object is designed to:
    * Store all the variables
    * Provide a list of variables
        * as symbols
        * as values
    * Rapidly set a variable value from a list
    * Provide the index of a named parameter
    * Manage variables e.g. duplicates

    Shares:
    * Namespace
    * Numeric values
    '''

    def __init__(
            self,
            variables=None,
            storage_type:str='variable', 
            acceptable_value_types:list=[float, int],
            acceptable_variable_classes:tuple=(str, ODEVariable)
        ):
        '''
        Intiialise an empty store
        '''

        # Dict of ODEVariable objects. key:value = ID:ODEVariable
        self._variables = IndexedOrderedDict()

        # Variable position for O(1) look up. key:value = ID:index
        self._variable_pos = dict()

        self.storage_type = storage_type
        self.index = IndexShim(parent=self)     # TODO: check what this is doing

        # Acceptable ways to define a variable
        self._acceptable_variable_classes = acceptable_variable_classes

        # Set up containers to store values, sorted by type
        acceptable_value_types.append(NoneType) # None (default) is always ok

        self._acceptable_value_types = {
            avt: avt.__name__ for avt in acceptable_value_types
        }
        self._values_by_type = {
            key: set() for key in self._acceptable_value_types.values()
        }

        self._invalidate_cache()

        if variables:
            self.add(variables)

    def __getitem__(self, item:str) -> ODEVariable:
        '''
        Getter when referencing the variable by name,
        returns the OdeVariable instance.
        '''
        return self._variables[item]

    def __len__(self)->int:
        '''
        The current number of variables in the store
        '''
        return len(self._variables)
    
    def __contains__(self, key:str) -> bool:
        '''
        Does the variable already exist in the store?
        '''
        if not isinstance(key, str):
            raise TypeError(
                f'{self.storage_type} IDs must be of str type, was {type(key)}.'
            )
        return key in self._variables

    #####################################
    # Add variables
    #####################################

    def _get_value_type(self, value):
        '''
        Get data type of value, returns None if not
        one of the accepted types.
        '''
        for at, atn in self._acceptable_value_types.items():
            if isinstance(value, at):
                return atn
        return None

    def _force_list(self, variable):
        """
        Convert var into [var] and leave [var1, var2, var3] as it is
        """
        if isinstance(variable, self._acceptable_variable_classes):
            return [variable]
        if isinstance(variable, list):
            return variable
        raise InputError(
            f"Variable must be type 'list' or {self._acceptable_variable_classes}"
        )

    @abstractmethod
    def _check_variable(
            self,
            variable:str|ODEVariable
        ) -> ODEVariable:
        '''
        Normalise any user-provided representation of a variable (or variables)
        into a list of ODEVariable objects

        Parameters
        ----------
        variable: str | ODEVariable
            String or ODEVariable representation of variable

        NOTE: Let's only allow string or ODEVariable definitions.
              String sets the name with defaults, ODEVaribale allows
              for extra settings.
        '''
        pass

    def _prepare_variables(
            self,
            variable:str|ODEVariable|list[str|ODEVariable],
            all_symbols:None|dict[str:Symbol]=None
        ):
        '''
        Prepare variables to be added to the store, by producing list of
        ODEVariables (if not already in this form).
        For derived parameters this will be overloaded to check the
        algebraic expressions.

        Parameters
        ----------
        variable: str|ODEVariable|list[str|ODEVariable]
            String or ODEVariable representation of variable
        all_symbols:
            Symbols declared in the entire (state+param+derived_param+time)
            namespace.
            Allows us to verify derived parameter expressions which may depend
            on symbols outside of it's own namespace, i.e. states and parameters.

        '''
        variable = self._force_list(variable)
        var_list = [self._check_variable(var) for var in variable]

        return var_list

    def _add_var(self, var_obj, all_symbols=None):
        """
        Add list of variables to the store
        """
        key = var_obj.ID
        # Self defence, IDs have to be a string
        if not isinstance(key, str):
            raise TypeError(
                f"{self.storage_type} IDs must be of str type, was '{type(key)}'."
            )

        if key in self._variables:
            raise InputError(
                f"You may not add a {self.storage_type} more "
                f"than once. '{key}' already exists."
            )

        if all_symbols is not None:
            if key in all_symbols:
                # Even though all_symbols contains the namespace of self, we 
                # have already found no coflict there, so any issue must be due
                # to the sibling namespaces
                raise InputError(
                    f"Variable name, '{key}', already exists in another namespace"
                )         

        # key is new, so we need to record it:
        # Store the new variable, NOTE: doesn't seem to be a mechanism to update a variable
        self._variables[key] = var_obj
        # Store the position of this key
        self._variable_pos[key] = len(self._variables) 

    def add(self, variable, all_symbols=dict()):
        '''
        User method to add variable(s) to the store

        NOTE: can be overloaded depending on value storage

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
            of the list of variables
        '''
        var_list = self._prepare_variables(variable, all_symbols)

        for var_obj in var_list:
            self._add_var(var_obj, all_symbols)

        # store contents have changed, invalidate cached objects
        self._invalidate_cache()

    # def remove()
    #   TODO: should be method to remove too?

    #####################################
    # Properties
    # TODO: check if we need to add / remove any of these
    #####################################

    def get_index(self, key:str) -> int:
        '''
        Get the index of a particular variable 
        
        This should be fast ~ O(1)
        '''
        return self._variable_pos[key]

    ## Lists ##

    @property
    def id_list(self)->list[str]:
        '''
        Get a list of string variable names in the order they were added
        '''
        if self._id_list is None:
            self._id_list = [variable.ID for variable in self._variables.values()]
        
        return self._id_list

    @property
    def symbol_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        if self._symbol_list is None:
            self._symbol_list = [variable.symbol for variable in self._variables.values()]
        
        return self._symbol_list

    @property
    def lower_limit_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        if self._lower_limit_list is None:
            self._lower_limit_list = [variable.limits[0] for variable in self._variables.values()]

        return self._lower_limit_list

    @property
    def upper_limit_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        if self._upper_limit_list is None:
            self._upper_limit_list = [variable.limits[1] for variable in self._variables.values()]

        return self._upper_limit_list

    @property
    def full_list(self)->list[float]:
        '''
        Get a list of all the variables stored as ODEVariable objects
        '''
        if self._full_list is None:
            self._full_list = [variable for variable in self._variables.values()]

        return self._full_list

    ## Dicts ## 

    @property
    def symbol_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of all the symbols stored, keyed on the str 
        representation and value equal to the symbol
        '''
        if self._symbol_dict is None:
            result = OrderedDict()
            for variable in self._variables.values():
                result[variable.ID] = variable.symbol
            self._symbol_dict = result

        return self._symbol_dict

    ## Sets ## 

    @property
    def id_namespace(self)->set[str]:
        '''
        Get the set of all the id's stored
        '''
        if self._id_namespace is None:
            self._id_namespace = set(self.id_list)
        
        return self._id_namespace

    @property
    def symbol_namespace(self)->set[Symbol]:
        '''
        Get the set of all the symbols stored
        '''
        if self._symbol_namespace is None:
            self._symbol_namespace = set(self.symbol_list)
        
        return self._symbol_namespace

    def _invalidate_cache(self):
        """ Clear variable cache """
        self._id_list = None
        self._symbol_list = None
        self._lower_limit_list = None
        self._upper_limit_list = None
        self._full_list = None
        self._symbol_dict = None
        self._id_namespace = None
        self._symbol_namespace = None

class StateStore(VariableStore):
    '''
    A class to store states of an ODE system.
    
    This is basically unchanged from the parent class.
    '''
    def __init__(self, variables=None)->None:
        super().__init__(
            variables=variables,
            storage_type='state',
            acceptable_value_types=[
                int,
                float
            ],
            acceptable_variable_classes = (
                str,
                State
            )
            # acceptable_tags = [
            #     "alive",
            #     "dead",
            #     "infected",
            #     "infectious",
            #     "cumulative"
            # ]
        )
        # self._realisation_vals = None

        # self._states_by_tag = {
        #     key: dict() for key in self._acceptable_tags
        # }

    def _check_variable(self, variable):
        if isinstance(variable, State):
            return variable
        elif isinstance(variable, str):
            return State(ID=variable)
        else:
            raise InputError(
                f'You may not add an object of type '
                f'{type(variable)} as a {self.storage_type}.'
            )

    # def _invalidate_cache(self):
    #     '''
    #     Overload parent class to include derived parameter specific properties to invalidate
    #     '''
    #     super()._invalidate_cache()
    #     self._initial_conditions_dict = None

    # def get_states_by_tag(self, tag):

    #     return [
    #         state.symbol
    #         for state in self._variables.values()
    #         if tag in state.tags
    #     ]

    def get_dynamic_states(self):

        return [
            state.symbol
            for state in self._variables.values()
            if "cumulative" not in state.tags
        ]

    # TODO: 

    # def initial_conditions(self):
    #     '''
        
    #     '''

    # @property
    # def all_initial_conditions_set(self)->bool:
    #     '''
    #     Have all the values been set?
    #     '''
        
class ParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    variables of the type CallableParameter
    '''
    def __init__(self, variables=None, rng=None)->None:
        super().__init__(
            variables=variables,
            storage_type='parameter',
            acceptable_value_types=[
                int,
                float,
                rv_frozen,
                CallableParameter
            ],
            acceptable_variable_classes = (
                str,
                Parameter
            )
        )

        self._invalidate_value_cache()

        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

    def _check_variable(self, variable):
        if isinstance(variable, Parameter):
            return variable
        elif isinstance(variable, str):
            return Parameter(ID=variable)
        else:
            raise InputError(
                f'You may not add an object of type '
                f'{type(variable)} as a {self.storage_type}.'
            )

    def _invalidate_value_cache(self):
        self._value_list = None
        self._id_value_dict = None
        self._symbol_value_dict = None

    def _add_var(self, var_obj, all_symbols=dict()):
        super()._add_var(var_obj, all_symbols)

        # deal with the bootstrapping problem (everything goes in None to start)
        self._values_by_type[NoneType.__name__].add(var_obj.ID)

    #####################################
    # Set numeric value of variables
    #####################################

    def new_realisation(self)->None:
        '''
        Generate the requirement for new realiasation of the parameters.
        This is stored in the Param objects
        Invalidate cache to let properties know they need to update
        '''
        # handle the different ways in which a stochastic parameter can get a 
        # new realisation
        for parameter in self._variables.values():
            if parameter.is_stochastic:
                parameter.realise(rng=self._rng)
        
        self._invalidate_value_cache()

    def _set_value(self, variable:str, value:Number|tuple) -> None:
        '''
        Set the numeric value of a variable

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The numeric value that the variable should take.
        '''

        # If variable does not exist, add it to the store first
        if variable not in self:
            self.add(variable)

        if isinstance(value, tuple):
            value = CallableParameter(value)

        current_value = self[variable].value
        current_type = self._get_value_type(current_value)

        new_type = self._get_value_type(value)

        if new_type is None:
            raise InputError(
                f'You may not add an object of type {type(value).__name__}'
                f' as a value for a {self.storage_type}.'
                f' Only {list(self.acceptable_value_types.keys())} are '
                'permitted (or sub-classes).')

        if current_type != new_type:
            # self._values_by_type[current_type].pop(variable, None)
            # Set a pointer to the new location
            # self._values_by_type[new_type][variable] = self._variables[variable]
            self._values_by_type[current_type].discard(variable)
            self._values_by_type[new_type].add(variable)

        # Set the value   
        self[variable].value = value

        self._invalidate_value_cache()

    def set_values(self, values:dict[str: float]) -> None:
        '''
        Set the value of the variables

        This is explicit and so the prefered way to set the variable values.

        Parameters
        ----------
        Values: A dict keyed on the variable name with value equal to the value.
        ''' 
        for key, value in values.items():
            self._set_value(key, value)

        # self._invalidate_value_cache()

    @property
    def value_list(self)->list[float]:
        '''
        Get a list of all the variables stored as ODEVariable objects
        '''
        if self._value_list is None:
            self._value_list = [variable.value for variable in self._variables.values()]

        return self._value_list

    @property
    def id_value_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of all the values stored, keyed on the str 
        representation and value equal to the numeric value
        '''
        if self._id_value_dict is None:
            result = OrderedDict()
            for variable in self._variables.values():
                result[variable.ID] = variable.value
            self._id_value_dict = result

        return self._id_value_dict

    @property
    def symbol_value_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of all the values stored, keyed on the str 
        representation and value equal to the numeric value
        '''
        if self._symbol_value_dict is None:
            result = OrderedDict()
            for variable in self._variables.values():
                result[variable.symbol] = variable.value
            self._symbol_value_dict = result

        return self._symbol_value_dict

    @property
    def all_values_set(self)->bool:
        '''
        Have all the values been set?
        '''
        return len(self._values_by_type[NoneType.__name__]) == 0

    @property
    def stochastic_parameters(self)->dict[str: rv_frozen]:
        '''
        Provides a set of stochastic parameter id's

        Returns
        -------
        Set of str IDs
        '''
        result = self._values_by_type[rv_frozen.__name__].copy()
        result.update(self._values_by_type[CallableParameter.__name__])
        return result

    @property
    def has_stochastic_parameters(self)->bool:
        '''
        Simple check to see if there are any stochastic parameters in the store
        '''
        return len(self.stochastic_parameters) != 0

    # @property
    # def has_stochastic_parameters(self)->bool:
    #     '''
    #     Simple check to see if there are any stochastic parameters in the store
    #     '''
    #     return (
    #         len(self._values_by_type.get(rv_frozen.__name__, {}))  + 
    #         len(self._values_by_type.get(CallableParameter.__name__, {}))
    #     ) != 0

    # @property
    # def stochastic_parameters(self)->dict[str: rv_frozen]:
    #     '''
    #     Provides a dict of stochastic parameters (i.e. ones where the variable)
    #     has been defined as an instance of rv_frozen.

    #     Returns
    #     -------
    #     Dict keyed on parameter name with value = the distribution
    #     '''
    #     result = self._values_by_type[rv_frozen.__name__].copy()
    #     result.update(self._values_by_type[CallableParameter.__name__])
    #     return result


class DerivedParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    variables of the type CallableParameter
    '''
    def __init__(self, variables=None)->None:
        super().__init__(
            variables=variables,
            storage_type='derived parameter',
            acceptable_value_types=[
                int,
                float
            ],
            acceptable_variable_classes = tuple
        )
        # self._realisation_vals = None

    def _invalidate_cache(self):
        '''
        Overload parent class to include derived parameter specific properties to invalidate
        '''
        super()._invalidate_cache()
        self._expression_dict = None
        self._id_expression_dict = None
        self._symbol_expression_dict = None

    def _prepare_variables(self, variable, all_symbols):
        '''
        Overload parent class to check the derived parameter expressions
        '''
        var_list = super()._prepare_variables(variable, all_symbols)

        for derived_param in var_list:
            derived_param.build_sympy_expression(all_symbols)

        return var_list

    def _check_variable(self, variable):
        """
        Perform derived parameter specific input checks
        """
        if isinstance(variable, DerivedParameter):
            if variable.string_expression:
                return variable
            else:
                raise InputError(
                    "Defining derived prameters via DerivedParameter type "
                    "should already include the string expression"
                )

        if isinstance(variable, tuple):
            if len(variable) != 2:
                raise InputError(
                    "Derived parameters should be (name, expression)"
                )

            name, expr = variable

            if isinstance(name, str):
                if isinstance(expr, str):
                    return DerivedParameter(ID=name, string_expression=expr)
                else:
                    raise InputError(f"Expression should be type 'str', got '{type(expr)}'")
            else:
                raise InputError(f"Name should be type 'str', got '{type(expr)}'")

        raise InputError(
            f"Cannot create derived parameter from {type(variable)}"
        )


    @property
    def id_expression_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of derived parameter sympy expressions, keyed on the str 
        representation
        '''
        if self._id_expression_dict is None:
            result = OrderedDict()

            for variable in self._variables.values():
                result[variable.ID] = variable.sympy_expression
        
            self._id_expression_dict = result

        return self._id_expression_dict

    @property
    def symbol_expression_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of derived parameter sympy expressions, keyed on the symbolic 
        representation
        '''
        if self._symbol_expression_dict is None:
            result = OrderedDict()

            for variable in self._variables.values():
                result[variable.symbol] = variable.sympy_expression
        
            self._symbol_expression_dict = result

        return self._symbol_expression_dict