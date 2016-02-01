class ArrayError(Exception):
    '''
    Expecting an array
    '''
    pass

class EstimateError(Exception):
    '''
    Issues in optimization procedure
    '''
    pass

class ExpressionErrror(Exception):
    '''
    Fail to compile the sympy expression
    '''
    pass

class InitializeError(Exception):
    '''
    Fail to initialize the ode
    '''
    pass


class InputError(Exception):
    '''
    As the name suggest... the type of input to the function or
    method is not of the expected types
    '''
    pass

class IntegrationError(Exception):
    '''
    As the name suggest...
    '''
    pass

class OutputError(Exception):
    '''
    As the name suggest... there are problems with output
    Usually a get method
    '''
    pass

class SimulationError(Exception):
    '''
    When we cannot perform simulations
    '''
    pass