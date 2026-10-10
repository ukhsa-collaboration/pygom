"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that contains a variable object for the ode

"""
from sympy.physics.units.quantities import Quantity
import sympy
import numpy as np

from ._model_errors import InputError


class ODEVariable(object):
    """
    A class that defines the variables in our ODE

    Parameters
    ----------
    ID: str
        identifier of the variable
    name: str, optional
        name of the variable in human readable format.
        Defaults to None, which then takes the ID as the name
    units: str, optional
        what unit the variable takes. Defaults to None.
    real: bool, optional
        if the variable can only be a real number, defaults to True
    """
    def __init__(self, 
                 ID:str, 
                 symbol:None|sympy.Symbol=None,
                 value:None|str=None,
                 units:None|Quantity=None,
                 real:bool=False,
                 limits:None|tuple=None      # default: assume compartments represent counts
                 ):
        self.ID = ID
        if symbol is None:
            # Create a symbol if we need to
            symbol = sympy.symbols(ID, real=real)
        self.symbol = symbol
        self.value = value
        self.units = units

        if limits is None:
            self.limits = (0, np.inf)
        else:
            self.limits = limits

    def __str__(self)->str:
        return self.ID

    def __repr__(self)->str:
        return (f'ODEVariable({repr(self.ID)}, '
                            f'{repr(self.symbol)}, '
                            f'{repr(self.value)}, '
                            f'{repr(self.units)}, '
                            f'{repr(self.limits)}')
                                                

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, ODEVariable):
            return self.ID == other.ID and \
                self.symbol == other.symbol and \
                self.units == other.units and \
                self.limits == other.limits
        elif isinstance(other, sympy.Symbol):
            return self.ID == str(other)
        else:
            raise NotImplementedError('Wrong input type of %s' % type(other))

    def __neq__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __le__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __gt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __ge__(self, other):
        raise NotImplementedError("Only equality comparison allowed")
    
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, value:sympy.Symbol):
        if not isinstance(value, sympy.Symbol):
            raise InputError('The symbol attribute must be a sympy Symbol ' 
                             'object.')
        self._symbol = value

