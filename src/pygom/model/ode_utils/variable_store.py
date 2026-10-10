'''
An object to hold variables for a PyGOM ODE system

This object is designed to:
* Store all the variables
* Provide a list of variables
  * as symbols
  * as values
* Rapidly set a variable value from a list
* Provide the index of a named parameter

'''
from types import NoneType
from collections import OrderedDict
from indexed import IndexedOrderedDict
import re
import warnings

from sympy import Symbol, symbols
import numpy as np

from scipy.stats._distn_infrastructure import rv_frozen

from ..ode_variable import ODEVariable
from .._model_errors import InputError

__all__ = ['VariableStore','ParameterStore', 'StateStore']

re_math = re.compile(r'[-+*\\]')
re_underscore = re.compile('^_')
re_symbol_name = re.compile('[A-Za-z_]+')

def _generate_symbol(input_str:str|tuple)->Symbol:
    '''
    Check the symbol name and turn it into a symbol 
    '''
    # What type of definition are we dealing with?
    if isinstance(input_str, (list, tuple)):
        if len(input_str) == 2:
            if str(input_str[1]).lower() in ("complex", "false"):
                is_real = 'False'
                symbol_name = input_str[0]
            elif str(input_str[1]).lower() in ("real", "true"):
                is_real = 'True'
                symbol_name = input_str[0]
            else:
                raise InputError("Unexpected second argument for symbol")
        else:
            raise InputError("Unexpected number of arguments for symbol")
    elif isinstance(input_str, str):  # assume real unless stated otherwise
        is_real = 'True'
        symbol_name = input_str
    else:
        raise InputError("Unexpected input type for symbol")

    #Some basic name checks
    if re_math.search(symbol_name) is not None:
        raise InputError('Mathematical operators not allowed in symbol '
                         'definition')
    if re_underscore.search(symbol_name) is not None:
        raise InputError('A symbol cannot have underscore as first character.')

    if symbol_name == 'lambda':
        raise InputError('lambda is a reserved keyword')

    tempSym = symbols(symbol_name, real=is_real)

    if isinstance(tempSym, Symbol):
        return tempSym
    elif isinstance(tempSym, tuple):
        if len(tempSym) == 0:
            raise InputError("Input symbol is not valid")
        return list(tempSym)
    else:
        raise InputError("Unexpected result using the input string:"
                             + str(tempSym))

class IndexShim(object):
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, item:int):
        return self.parent._variables.values()[item]

class VariableStore(object):
    def __init__(self, 
                 storage_type:str='variable', 
                 acceptable_value_types:list=[float, int]):
        '''
        The init method
        '''
        self._variables = IndexedOrderedDict()
        self._variable_pos = dict()
        self.storage_type = storage_type
        self.index = IndexShim(parent=self)

        # Type checking for the values assigned to a parameter
        acceptable_value_types.append(NoneType) # None is always ok
        self.acceptable_value_types = {avt: avt.__name__ for avt in 
                                       acceptable_value_types}
        self._values_by_type = {key: dict() for key in 
                                self.acceptable_value_types.values()}

        #self.sibling_lists = []

    def __getitem__(self, item:str):
        '''
        Getter when referencing the variable by name
        '''
        return self._variables[item]

    def __setitem__(self, key:str, value:str|Symbol|ODEVariable) -> None:
        '''
        Setter when referencing by name
        '''
        # Self defence, IDs have to be a string
        if not isinstance(key, str):
            raise TypeError(f'{self.storage_type} IDs must be of str type, was'
                            f'{type(key)}.')
        
        # convert the value to an ODEVariable
        var_list:list[ODEVariable] = self._check_variable(variable=value)

        for var_obj in var_list:
            #TODO: Check that this new variable is not in the sibling lists

            # check to see if we need to record the position of this key
            # and record it if we do
            if key not in self._variables:
                self._variable_pos[key] = len(self._variables) 

            # Store the new / updated variable
            self._variables[key] = var_obj

            # re-record the value
            value = var_obj.value

            # deal with the bootstrapping problem (everything goes in None to start)
            self._values_by_type[NoneType.__name__] [key]=self._variables[key]

            # Properly log the real value (maintains book-keeping)
            self.set_value(key, value)

    def __len__(self)->int:
        '''
        The current length of the store
        '''
        return len(self._variables)
    
    def __str__(self)->str:
        '''
        gives a string representation of the parameters
        '''
        return ', '.join(self._variables.keys())
    
    def __contains__(self, key)->bool:
        if not isinstance(key, str):
            raise TypeError(f'{self.storage_type} IDs must be of str type, was'
                            f'{type(key)}.')
        return key in self._variables

    def get_index(self, key:str)->int:
        '''
        Get the index of a particular variable 
        
        This should be fast - O(1)
        '''
        return (self._variable_pos[key])

    def append(self, 
               variable:str|Symbol|ODEVariable, 
               symbol:Symbol|None=None, 
               real:bool=True) -> ODEVariable:
        '''
        Add a variable to the store

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
          of the list of variables

        '''
        var_list:list[ODEVariable] = self._check_variable(variable=variable,
                                                          symbol=symbol,
                                                          real=real
                                                          )

        for var_obj in var_list:
            if var_obj.ID in self._variables:
                raise InputError(f'You may not add a {self.storage_type} more '
                                    f'than once. {var_obj.ID} already exists.'
                                    )
            # Store the new variable
            self[var_obj.ID] = var_obj
    
    def extend(self, variables:list) -> None:
        '''
        Add a list of variables to the store

        Parameters
        ----------
        variables: A list of variables to add, either as a list of strings or
        ODEVariables. These will be appended at the end of the list of variables

        '''
        for variable in variables:
            # Quick / safe way to generate the ID although we will "check" twice
            var_list:list[ODEVariable] = self._check_variable(variable=variable)
            for var_obj in var_list:
                self[var_obj.ID] = var_obj
    
    def _check_variable(self, 
                        variable:str|Symbol|ODEVariable,
                        symbol:Symbol|None=None, 
                        real:bool=True,
                        limits:tuple|None=None)->list[ODEVariable]:
        '''
        Turn variable into a list of ODEVariables

        This will often retun a list of length 1 but because of the way the 
        string conversion works it may be longer. E.g. a string 'y1:4' will
        result in a list of ODEVariables representing [y1, y2, y3, y4].
        '''
            
        # TODO: Surface the units part of ODEVariable
        if isinstance(variable, str):
            # Check what ID / symbol we are going to use
            if symbol is not None:
                warnings.warn(f'Variable was a string and symbol was set. Will '
                              f' using ID given {variable} with symbol '
                              f'{symbol}')
                symbols = symbol
            else:
                symbols=_generate_symbol(variable)

            # did the conversion result in one or more symbols?
            # TODO: if a list i don't think thiw will work
            if isinstance(symbols, list):
                var_obj = [ODEVariable(ID=str(symbol),
                                       symbol=symbol,
                                       real=real,
                                       limits=limits)
                           for symbol in symbols]
            else:
                var_obj = [ODEVariable(ID=variable, 
                                       symbol=symbol,
                                       real=real,
                                       limits=limits)]
                
        elif isinstance(variable, Symbol):
            var_obj = [ODEVariable(ID=str(variable),
                                   symbol=variable,
                                   real=real,
                                   limits=limits)]
        elif isinstance(variable, ODEVariable):
            var_obj = [variable]
        else:
            raise InputError(f'You may not add an object of type '
                             f'{type(variable)} as a {self.storage_type}.')
        
        return var_obj



    def set_value(self, variable:str, value) -> None:
        '''
        Set the value of a variable

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The value that the variable should take.

        '''
        # Book-keeping for the by-type dicts
        current_type = ''
        for at, atn in self.acceptable_value_types.items():
            if isinstance(self[variable].value, at):
                current_type = atn

        new_type = ''
        for at, atn in self.acceptable_value_types.items():
            if isinstance(value, at):
                new_type = atn
        

        if new_type != current_type:
            self._values_by_type[current_type].pop(variable, None)
           
            if new_type == '':
                raise InputError(f'You may not add an object of type {type(value).__name__}'
                                 f' as a value for a {self.storage_type}.'
                                 f' Only {list(self.acceptable_value_types.keys())} are '
                                 'permitted (or sub-classes).')
            
            # Set a pointer to the new location
            self._values_by_type[new_type] [variable]=self._variables[variable]

        # Set the value   
        self[variable].value = value
    
    def set_value_list(self, values:list) -> None:
        '''
        Set the value of all the variables

        Parameters
        ----------
        Values: A list, the same length as the number of variables, 
          containing the values 
        '''
        if len(values) != len(self):
            raise ValueError(F'The length of the supplied list of values must '
                             f'match the number of {self.storage_type}. '
                             f'Expected {len(self)}, got {len(values)}.')
        
        for key, value in zip(self._variables.keys(), values):
            self.set_value(key, value)
        
    def set_value_dict(self, values:dict):
        '''
        Set the value of the variables

        This is explicit and so the prefered way to set the variable values.

        Parameters
        ----------
        Values: A dict keyed on the variable name with value equal to the value.
        ''' 
        for key, value in values.items():
            self.set_value(key, value)


    @property
    def all_values_set(self)->bool:
        '''
        Have all the values been set?
        '''
        return len(self._values_by_type[NoneType.__name__]) == 0
    
    @property
    def variables(self)->list[str]:
        '''
        Get a list of strings of the names for all the variables 
        '''
        return [variable.ID for variable in self._variables.values()]

    @property
    def values(self)->list[float]:
        '''
        Get a list of all the values stored
        '''
        return [variable.value for variable in self._variables.values()]
    
    @property
    def values_full(self)->list[float]:
        '''
        Get a list of all the values stored as ODEVariable objects
        '''
        return [variable for variable in self._variables.values()]
    
    @values.setter
    def values(self,
               values:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        '''
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
        '''
        # Either a list or a dict
        if isinstance(values, (list, tuple, np.ndarray)):
            # Looks like a list but is it a dict in disguise (list of tuples)?
            if len(values) > 0:
                if isinstance(values[0], tuple) and len(values[0]) == 2:
                    # do we have at least one tuple of length 2?
                    try:
                        values = {key: value for key, value in values}
                    except ValueError as e:
                        raise ValueError(f'The {self.storage_type} list' 
                                         ' supplied looked like a list of'
                                         ' tuples, (NAME, VALUE) and'
                                         ' PyGOM tried to evaluate it on that'
                                         ' basis but these entries '
                                        f'{[value for value in values if len(value)!=2]},'
                                        ' were not of length 2,'
                                        ' please check these.') from e
                    # Set as dict
                    self.set_value_dict(values)
                else:
                    # Not a dict in disguise, set as a list
                    self.set_value_list(values)
        elif isinstance(values, dict):
            # This is the way, a eplicit dict of [ID: value]
            self.set_value_dict(values)
        else:
            raise InputError(f'Expecting a dict, or iterable '
                             f'input not {type(values)}')
    @property
    def symbol_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        return [variable.symbol for variable in self._variables.values()]
    
    @property
    def symbol_dict(self)->dict[str: Symbol]:
        '''
        Get a OrderedDict of all the symbols stored, keyed on the str 
        representation and value equal to the symbol
        '''
        result = OrderedDict()

        for variable in self._variables.values():
            result[variable.ID] = variable.symbol
        return result

# class CallableParameter(object):
#     '''A class to wrap a parameter supplied as a callable '''
#     def __init__(self, value:tuple):
#         '''
#         parameters
#         ----------
#         value: tuple in either (callable, (paramerters....)) or 
#           (callable, {parameters})
#         '''
        
#         if not callable(value[0]):
#             raise InputError('First element should be a callable when using '
#                              'multi argument distribution definition.  Type of '
#                              f'input was {type(value[0])}.')
#         self._callable = value[0] 

#         # Now deal with the parameters
#         if isinstance(value[1], dict):
#             self.kwargs = value[1]
#             self.args = []
#         elif isinstance(value[1], tuple):
#             self.kwargs = {}
#             self.args = value[1]
#         else:
#             raise InputError('Second element should be either a tuple or a '
#                              'dict when using multi-argument distribution '
#                              f'definition. Type of input was {type(value[1])}.')
#     def __call__(self, n=1):
#         return self._callable(n, *self.args, **self.kwargs,)
    
class CallableParameter:
    def __init__(self, value: tuple, rng=None):
        if not callable(value[0]):
            raise InputError("First element must be callable.")
        self._callable = value[0]

        # parse args/kwargs
        if isinstance(value[1], dict):
            self.args = []
            self.kwargs = value[1]
        elif isinstance(value[1], tuple):
            self.args = value[1]
            self.kwargs = {}
        else:
            raise InputError('Second element should be either a tuple or a '
                             'dict when using multi-argument distribution '
                             f'definition. Type of input was {type(value[1])}.')

        # TODO: validate method to store rng
        self.rng = rng

    def __call__(self, n=1):
        """
        Call the underlying function.

        If the function accepts an `rng` argument, pass it.
        Otherwise fall back to the old behavior.
        """

        return self._callable(n, *self.args, rng=self.rng, **self.kwargs)
        # try:
        #     # Try passing rng explicitly
        #     return self._callable(n, *self.args, rng=self.rng, **self.kwargs)
        # except TypeError:
        #     # Function did not accept rng → backwards compatible path
        #     return self._callable(n, *self.args, **self.kwargs)

class ParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    values of a parameter that are draws from a stochatic distribution.
    '''
    def __init__(self)->None:
        super().__init__(storage_type='parameter',
                         acceptable_value_types=[int,
                                                 float, 
                                                 rv_frozen,
                                                 CallableParameter
                                                 ]
                         )
        self._realisation_vals = None

    def set_value(self, variable, value):
        '''
        Sets the value of a variable
        '''
        # convert callables nested in tuples into callables class
        if isinstance(value, tuple):
            value = CallableParameter(value)

        return super().set_value(variable, value)
        
    @property
    def has_stochastic_parameters(self)->bool:
        '''
        Simple check to see if there are any stochastic parameters in the store
        '''
        return (len(self._values_by_type.get(rv_frozen.__name__, {}))  + 
                len(self._values_by_type.get(CallableParameter.__name__, {}))) != 0
        
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
        
        If there are stochastic parameters then a draw will be made and stored
        and returned on subsequent calls to this method.

        To generate a new realisation call new_realisation.

        Returns
        -------
        A list of numeric values.

        For each element in the list if a parameter isstochastic then a new 
        value is drawn, if it is deterministic then the value is simply added. 
        '''
        # Check the cache for an existing set (and return that )
        if self._realisation_vals is not None:
            return self._realisation_vals
        
        # Build a new parameter set
        result = list()

        # handle the different ways in which a stochastic parameter can get a 
        # new realisation
        for parameter in self._variables.values():
            if isinstance(parameter.value, rv_frozen):
                result.append(parameter.value.rvs(1, random_state=self.rng)[0])
            elif isinstance(parameter.value, CallableParameter):
                result.append(parameter.value())
            else:
                # The deterministic case
                result.append(parameter.value)
        
        #cache the result
        self._realisation_vals = result
        
        return result
    
    @values.setter
    def values(self,
               values:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        # set the values via the parent property
        VariableStore.values.fset(self, values)

        # Reset the cache
        self._realisation_vals = None

class StateStore(VariableStore):
    '''
    A class to store parameters of an ODE system
    '''
    def __init__(self)->None:
        super().__init__(storage_type='state',
                         acceptable_value_types=[int,
                                                 float, 
                                                 ]
                         )
        self._realisation_vals = None

    def _check_variable(self, 
                        variable:str|Symbol|ODEVariable,
                        symbol:Symbol|None=None, 
                        real:bool=True,
                        limits:tuple|None=None)->ODEVariable:

        limits = (0, np.inf)

        # we expect a state defining tuple to be in the form:
        # ('NAME', (MIN, MAX)). Test this then create a suitable symbol
        if isinstance(variable, tuple):
            if len(variable) != 2:
                raise InputError("Variable must be tuple of length 2")
            else:
                if not isinstance(variable[0], str):
                    raise InputError("Variable must be of type string")
                elif len(variable[0].strip()) == 0:
                    raise InputError("Variable has no name")
                elif not isinstance(variable[1], tuple):
                    raise InputError("Limits must be type tuple")
                elif len(variable[1])!=2:
                    raise InputError("Limit tuple must be length 2")
                else:
                    limits = variable[1]
                    variable = variable[0]

            low  = limits[0] if limits[0] is not None else 0
            high = limits[1] if limits[1] is not None else np.inf

            limits = (low, high)

        results = super()._check_variable(
            variable=variable,
            symbol=symbol,
            real=real,
            limits=limits
            )

        return results
        
    