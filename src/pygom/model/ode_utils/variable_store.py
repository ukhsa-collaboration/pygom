from types import NoneType
from collections import OrderedDict
from indexed import IndexedOrderedDict

import numpy as np

from sympy import Symbol, symbols

from pygom.model.ode_variable import ODEVariable, State, Parameter, DerivedParameter, CallableParameter
from pygom.model._model_errors import InputError

from scipy.stats._distn_infrastructure import rv_frozen

from .._model_verification import checkEquation

from abc import abstractmethod

__all__ = ['VariableStore','ParameterStore', 'StateStore', 'DerivedParameterStore']

"""
Inheritance hierarchy may still be useful. 
Forcing states and params to expose identical value-management behaviour
might be making design awkward
"""


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
            storage_type:str='variable', 
            acceptable_value_types:list=[float, int],
            acceptable_variable_classes:tuple=(str, Symbol, ODEVariable)
        ):
        '''
        Intiialise an empty store
        '''

        # Dict of ODEVariable objects. key:value = id:ODEVariable
        self._variables = IndexedOrderedDict()

        # Variable position for O(1) look up
        self._variable_pos = dict()

        self.storage_type = storage_type
        self.index = IndexShim(parent=self)

        self._acceptable_variable_classes = acceptable_variable_classes

        # Set up containers to store values, sorted by type

        acceptable_value_types.append(NoneType) # None (default) is always ok

        self._acceptable_value_types = {
            avt: avt.__name__ for avt in acceptable_value_types
        }
        self._values_by_type = {
            key: dict() for key in self._acceptable_value_types.values()
        }

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
            f"Variable must be list or {self._acceptable_variable_classes}"
        )

    @abstractmethod
    def _check_variable(
            self,
            variable:str|Symbol|ODEVariable
        ) -> ODEVariable:
        '''
        Normalise any user-provided representation of a variable (or variables)
        into a list of ODEVariable objects

        Parameters
        ----------
        variable: str | sympy.Symbol | ODEVariable
            String, symbolic or ODEVariable representation of variable
        symbol: sympy.Symbol (optional)
            Symbolic representation of variable
        real: bool
            True if variable is real
        limits: tuple[number, number]
            Minimum and maximum allowed values.

        TODO:
            1) why allow users to specify string, sympy symbols or ODEvars?
            seems like too many options that makes this bit awkward 
            Do we want users bringing sympy objects into pygom themselves?
        '''
        pass

    def _prepare_variables(self, variable, sibling_namespace=None):
        '''
        Prepare variables to be added by producing list of
        ODEVariables. For derived parameters this will be
        overloaded to check the expressions.
        '''
        variable = self._force_list(variable)
        var_list = [self._check_variable(var) for var in variable]

        return var_list

    def _add_var_list(self, var_list, sibling_namespace=None):
        """
        Add list of variables to the store
        """
        for var_obj in var_list:
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

            if sibling_namespace is not None:
                if key in sibling_namespace:
                    raise InputError(
                        f"Variable name, '{key}', already exists in another namespace"
                    )         

            # check to see if we need to record the position of this key
            # and record it if we do
            if key not in self._variables:
                self._variable_pos[key] = len(self._variables) 

            # Store the new / updated variable
            self._variables[key] = var_obj

    def add(
            self, 
            variable,
            sibling_namespace=dict()
        ):
        '''
        User method to add variable(s) to the store

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
            of the list of variables
        '''
        var_list = self._prepare_variables(variable, sibling_namespace)
        self._add_var_list(var_list, sibling_namespace)

    # def remove()
    #   TODO: should be method to remove too

    #####################################
    # Set numeric value of variables
    #####################################

    def _set_value(self, variable:str, value) -> None:
        '''
        Set the numeric value of a variable

        TODO: 
        1) does the variable need to exist already?

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The numeric value that the variable should take.
        '''

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
            self._values_by_type[current_type].pop(variable, None)
            # Set a pointer to the new location
            self._values_by_type[new_type][variable] = self._variables[variable]

        # Set the value   
        self[variable].value = value

    @property
    def values(self)->list[float]:
        '''
        Get a list of all the numerical values stored
        '''
        return [variable.value for variable in self._variables.values()]

    @values.setter
    def values(
        self,
        values:dict[str: float]
    ) -> None:
        '''
        Set the value of the variables

        This is explicit and so the prefered way to set the variable values.

        Parameters
        ----------
        Values: A dict keyed on the variable name with value equal to the value.
        ''' 
        for key, value in values.items():
            self._set_value(key, value)

    #####################################
    # Useful shared attributes
    # TODO: maybe not all required
    #####################################

    def get_index(self, key:str) -> int:
        '''
        Get the index of a particular variable 
        
        This should be fast - O(1)
        '''
        return self._variable_pos[key]

    @property
    def all_values_set(self)->bool:
        '''
        Have all the values been set?
        '''
        return len(self._values_by_type[NoneType.__name__]) == 0
    
    @property
    def variables(self)->list[str]:
        '''
        Get a list of string variable names in the order they were added
        '''
        return [variable.ID for variable in self._variables.values()]

    @property
    def variable_list(self)->list[str]:
        '''
        Get a list of string variable names in the order they were added
        '''
        return [variable.ID for variable in self._variables.values()]

    @property
    def id_namespace(self)->list[str]:
        '''
        Get the set of variable names
        '''
        return set([variable.ID for variable in self._variables.values()])

    @property
    def symbol_namespace(self)->list[str]:
        '''
        Get the set of variable symbols
        '''
        return set([variable.symbol for variable in self._variables.values()])

    @property
    def symbol_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        return [variable.symbol for variable in self._variables.values()]
    
    @property
    def symbol_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of all the symbols stored, keyed on the str 
        representation and value equal to the symbol
        '''
        result = OrderedDict()

        for variable in self._variables.values():
            result[variable.ID] = variable.symbol
        return result

    @property
    def lower_limit_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        return [variable.limits[0] for variable in self._variables.values()]

    @property
    def upper_limit_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        return [variable.limits[1] for variable in self._variables.values()]

    @property
    def values_full(self)->list[float]:
        '''
        Get a list of all the variables stored as ODEVariable objects
        '''
        return [variable for variable in self._variables.values()]

class StateStore(VariableStore):
    '''
    A class to store states of an ODE system.
    
    This is basically unchanged from the parent class.
    '''
    def __init__(self)->None:
        super().__init__(
            # variable=variable,
            storage_type='state',
            acceptable_value_types=[
                int,
                float
            ],
            acceptable_variable_classes = (
                str,
                Symbol,
                State
            )
        )
        self._realisation_vals = None

    def _check_variable(self, variable):
        if isinstance(variable, State):
            return variable
        elif isinstance(variable, str):
            return State(ID=variable)
        elif isinstance(variable, Symbol):
            return State(symbol=variable)
        else:
            raise InputError(
                f'You may not add an object of type '
                f'{type(variable)} as a {self.storage_type}.'
            )

class ParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    variables of the type CallableParameter
    '''
    def __init__(self, rng=None)->None:
        super().__init__(
            # variable=variable,
            storage_type='parameter',
            acceptable_value_types=[
                int,
                float,
                rv_frozen,
                CallableParameter
            ],
            acceptable_variable_classes = (
                str,
                Symbol,
                Parameter
            )
        )

        # Cache status (None -> new params, if stochastic, need to be generated)
        self._realisation_vals = None

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def _check_variable(self, variable):
        if isinstance(variable, Parameter):
            return variable
        elif isinstance(variable, str):
            return Parameter(ID=variable)
        elif isinstance(variable, Symbol):
            return Parameter(symbol=variable)
        else:
            raise InputError(
                f'You may not add an object of type '
                f'{type(variable)} as a {self.storage_type}.'
            )

    def _set_value(self, variable, value):
        '''
        Sets the value of a variable
        '''
        # convert callables nested in tuples into callables class
        if isinstance(value, tuple):
            value = CallableParameter(value)

        return super()._set_value(variable, value)
        
    @property
    def has_stochastic_parameters(self)->bool:
        '''
        Simple check to see if there are any stochastic parameters in the store
        '''
        return (
            len(self._values_by_type.get(rv_frozen.__name__, {}))  + 
            len(self._values_by_type.get(CallableParameter.__name__, {}))
        ) != 0
        
    @property
    def stochastic_parameters(self)->dict[str: rv_frozen]:
        '''
        Provides a dict of stochastic parameters (i.e. ones where the variable)
        has been defined as an instance of rv_frozen.

        Returns
        -------
        Dict keyed on parameter name with value = the distribution
        '''
        result = self._values_by_type[rv_frozen.__name__].copy()
        result.update(self._values_by_type[CallableParameter.__name__])
        return result
    
    def new_realisation(self)->None:
        '''
        Generate a new realiasation of the parameters
        '''
        # Just wipe the cache
        self._realisation_vals = None

    @property
    def values(self)->list[float]:
        '''
        Provides the values for the parameters
        
        Overload the parent method because there may be stochastic params.
        If there are stochastic parameters then a draw will be made and stored
        and returned on subsequent calls to this method.

        To generate a new realisation call new_realisation.

        Returns
        -------
        A list of numeric values.

        For each element in the list if a parameter isstochastic then a new 
        value is drawn, if it is deterministic then the value is simply added. 
        '''
        # Check the cache for an existing set (and return that)
        if self._realisation_vals is not None:
            return self._realisation_vals
        
        # Build a new parameter set
        result = list()

        # handle the different ways in which a stochastic parameter can get a 
        # new realisation
        for parameter in self._variables.values():
            if parameter.is_stochastic:
                parameter.realise(rng=self.rng)
            result.append(parameter.value)
        
        #cache the result
        self._realisation_vals = result
        
        return result
    
    @values.setter
    def values(
        self,
        values:dict[str: float]|list[tuple[str,float]]|list[float]
    ) -> None:
        # set the values via the parent property
        VariableStore.values.fset(self, values)

        # Reset the cache
        self._realisation_vals = None

class DerivedParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    variables of the type CallableParameter
    '''
    def __init__(self)->None:
        super().__init__(
            # variable=variable,
            storage_type='derived parameter',
            acceptable_value_types=[
                int,
                float
            ],
            accepted_variable_types = tuple
        )
        self._realisation_vals = None

    @property
    def expression_dict(self)->dict[str: Symbol]:
        '''
        Get an OrderedDict of all the symbols stored, keyed on the str 
        representation and value equal to the symbol
        '''
        result = OrderedDict()

        for variable in self._variables.values():
            result[variable.ID] = variable.sympy_expression
        return result

    def _prepare_variables(self, variable, sibling_namespace):
        '''
        Overload parent class to check the derived parameter expressions
        '''
        var_list = super()._prepare_variables(variable, sibling_namespace)

        for dp in var_list:
            dp.sympy_expression = checkEquation(
                dp.string_expression,
                sibling_namespace
                # self.expression_dict,
                # subs_derived=True,
            )

        return var_list

    def _check_variable(self, variable):
        """
        Perform derived parameter specific checks
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
                return DerivedParameter(ID=name, string_expression=expr)
            elif isinstance(name, Symbol):
                return DerivedParameter(symbol=name, string_expression=expr)

        raise InputError(
            f"Cannot create derived parameter from {type(variable)}"
        )