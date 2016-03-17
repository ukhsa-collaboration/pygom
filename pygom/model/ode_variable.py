import sympy

class ODEVariable(object):
    def __init__(self, ID, name=None, complex=False):
        self.ID = ID
        if name is None:
            self.name = ID
        else:
            self.name = name
        self.complex = complex
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return 'ODEVariable(%s, %s)' % (repr(self.ID), repr(self.name))
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, ODEVariable):
            return self.ID == other.ID and self.name == other.name
        elif isinstance(other, sympy.Symbol):
            return self.ID == str(other)
        else:
            raise NotImplementedError('Input type is %s, not an allowed type' % type(other))
        
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