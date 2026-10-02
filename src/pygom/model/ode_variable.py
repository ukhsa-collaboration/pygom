"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that contains a variable object for the ode

"""
from sympy.physics.units.quantities import Quantity
from sympy import symbols, Symbol
import numpy as np
import re
import keyword
from ._model_errors import InputError
from numbers import Number
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Callable

class CallableParameter:
    def __init__(self, value: tuple[Callable|str, dict|tuple]):
        """
        Data type for parameters which draws a random number when called

        Parameters
        ----------
        value: tuple[callable, dict|tuple]
            value[0] is the probability distribution and value[1] the function parameters
        """

        dist, params = value
        self.source = dist

        # -----------------------------
        # Parse arguments
        # -----------------------------
        if isinstance(params, dict):
            self.args = ()
            self.kwargs = params
        elif isinstance(params, tuple):
            self.args = params
            self.kwargs = {}
        else:
            raise InputError(
                'Second element should be either a tuple or a '
                'dict when using multi-argument distribution '
                f'definition. Type of input was {type(params)}.'
            )

    def __call__(self, rng):
        if isinstance(self.source, str):
            method = getattr(rng, self.source)

            return method(
                *self.args,
                **self.kwargs
            )

        return self.source(
            rng,
            *self.args,
            **self.kwargs
        )

class ODEVariable(object):
    """
    A class that defines the variable meta-data

    NOTE: Currently trialling not considering value as metadata

    Parameters
    ----------
    ID: str
        identifier of the variable
    symbol: sympy.Symbol
        sympy symbolic representation of the variable. Often taken to
        be sympy.Symbol(ID), but not necessarily.
    units: str, optional
        what unit the variable takes. Defaults to None.
    real: bool, optional
        if the variable can only be a real number, defaults to True
    limits: tuple(float, float), optional
        minimum and maximum allowed numerical values of the variable
    """
    def __init__(
            self,
            ID:None|str=None, 
            symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(0, np.inf),
            value=None
        ):

        if (ID is None) and (symbol is None):
            raise InputError(
                f"Must specify at least one of ID or symbol"
            )

        if ID is None:
            ID = str(symbol)

        if not isinstance(ID, str):
            raise TypeError("ID must be a string")
        self.ID = ID
        self.real = real
        self.units = units
        self.limits = limits
        self.value = value

        if symbol is None:
            symbol = ID
        self.symbol = symbol
        
    def __str__(self)->str:
        return self.ID

    def __repr__(self)->str:
        return (
            f"ODEVariable("         # TODO: change
            f"{self.ID!r}, "
            f"{self.symbol!r}, "
            f"{self.units!r}, "
            f"{self.limits!r})"
        )
                                                
    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, Symbol):
            return self.symbol == other
        elif isinstance(other, ODEVariable):
            return (
                self.ID == other.ID and \
                self.symbol == other.symbol
            )
        else:
            # raise NotImplementedError('Wrong input type of %s' % type(other))
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __le__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __gt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __ge__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def _generate_symbol(
            self,
            symbol_name: str,
            real:bool=True
        ) -> list:
        """
        Wrapper of sympy.symbols()

        We cannot let sympy build symbols on its own since we have some additional requirements.

        Generate one or more Sympy symbols from variable name(s)

        Parameters
        ----------
        symbol_name: str
            Name of the symbol
        real: bool
            True if real valued

        Returns
        -------
        sympy.Symbol
        """

        _SYMBOL_RULES = (
            "Symbol names must:\n"
            "  - start with a letter\n"
            "  - then contain only letters, digits or underscores"
            # "  - or be in SymPy range notation (e.g. 'y1:4')"
        )

        _VALID_SYMBOL = re.compile(
            r"^[A-Za-z][A-Za-z0-9_]*$"
        )

        # NOTE: should we make 't' a protected symbol for time?
        #       maybe add t at model initialisation and then any
        #       later attempts to add t will be blocked?
        if keyword.iskeyword(symbol_name):
            raise InputError(
                f"'{symbol_name}' is a reserved Python keyword"
            )
        if not _VALID_SYMBOL.fullmatch(symbol_name):
            raise InputError(
                f"Invalid symbol name '{symbol_name}'.\n{_SYMBOL_RULES}"
            )

        return symbols(symbol_name, real=real)

    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol:Symbol|str):
        if isinstance(symbol, str):
            symbol = self._generate_symbol(symbol, self.real)
        elif not isinstance(symbol, Symbol):
            raise InputError(
                'The symbol attribute must be of sympy.Symbol or str type'
            )
        self._symbol = symbol

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        if not isinstance(limits, (tuple, list)):
            raise InputError("Limits must be a tuple or list")

        if len(limits) != 2:
            raise InputError(f"Limits should contain exactly 2 values, received {len(limits)}")

        lower, upper = limits

        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise InputError("Limits must be numeric")

        if lower >= upper:
            raise InputError("Lower limit must be strictly less than upper limit")

        self._limits = (lower, upper)

    @property
    def value(self):
        return self._value

    # @value.setter
    # def value(self, value):
    #     self._source = value
    #     if isinstance(value, (rv_frozen, CallableParameter)):
    #         self._value = None
    #     else:
    #         self._validate_value(value)
    #         self._value = value

    @value.setter
    def value(self, value):

        if isinstance(value, (rv_frozen, CallableParameter)):
            self._source = value
            self._value = None

        else:
            self._validate_value(value)
            self._value = value


    def _validate_value(self, value):
        """
        Validate if a numerical value:
        - Is real if required
        - Falls within allowed limits
        """

        if value is None:
            return
        if isinstance(value, CallableParameter):
            return

        if self.real and not np.isreal(value):
            raise ValueError(
                f"'{self.ID}' must be real."
            )

        lower, upper = self.limits

        if value < lower:
            raise ValueError(
                f"'{self.ID}' must be >= {lower}."
            )

        if value > upper:
            raise ValueError(
                f"'{self.ID}' must be <= {upper}."
            )

class State(ODEVariable):
    def __init__(
            self,
            ID:None|str=None, 
            symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(0, np.inf),
            value:None|Number=None,
            initial_value:None|Number=None
        ):
        """
        If this object holds any numerical value then it refers to initial values.
        The solver ....
        """

        super().__init__(
            ID=ID,
            symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            value=value
        )

        self._initial_value = initial_value

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, value):
        self._validate_value(value)
        self._initial_value = value


class Parameter(ODEVariable):
    def __init__(
            self,
            ID:None|str=None, 
            symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(-np.inf, np.inf),
            value:None|Number|rv_frozen|CallableParameter=None
        ):

        self._source = value

        super().__init__(
            ID=ID,
            symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            value=value
        )

    # def realise(self, rng=None):
    #     """
    #     Generate a new parameter
    #     """

    #     if not self.is_stochastic:
    #         return self.value

    #     self.value = self._source(rng)

    #     return self.value

    def realise(self, rng=None):
        """
        Generate a new parameter
        """

        if not self.is_stochastic:
            return self.value

        new_value = self._source(rng)

        self._validate_value(new_value)
        self._value = new_value

        return new_value

    @property
    def is_stochastic(self):
        return callable(self._source)


class DerivedParameter(ODEVariable):
    def __init__(
            self,
            ID:None|str=None, 
            symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(-np.inf, np.inf),
            string_expression:None|str=None,
            value=None
        ):
        """

        """

        super().__init__(
            ID=ID,
            symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            value=value
        )

        self.string_expression = string_expression

    @property
    def string_expression(self):
        return self._string_expression

    @string_expression.setter
    def string_expression(self, eqn):
        """
        This needs to be checked and sympy-ed later on
        We do so when adding to the store
        Store owns each namespace, ModelSpec owns them all
        """
        if eqn is None:
            self._string_expression = None
        elif isinstance(eqn, str):
            self._string_expression = eqn
        else:
            raise(
                InputError(
                    "Derived parameter expression must be type 'str',"
                    f"instead received '{type(eqn)}'"
                )
            )
