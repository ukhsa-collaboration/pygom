"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that contains a variable object for the ode

"""

import sympy


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
    def __init__(self, ID, name=None, units=None, real=True):
        self.ID = ID
        if name is None:
            self.name = ID
        else:
            self.name = name
        self.units = units
        self.real = real

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'ODEVariable(%s, %s, %s, %s)' % (
                                                repr(self.ID),
                                                repr(self.name),
                                                repr(self.units),
                                                repr(self.real)
                                                )

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, ODEVariable):
            return self.ID == other.ID and \
                self.name == other.name and \
                self.units == other.units
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
