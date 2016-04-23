from libsbml import Parameter
from ._generic import getInfoFromList

def getParametersInfo(parameters):
    '''
    Return information for a list of parameters
    
    Parameters
    ----------
    parameters: list of :class:`libsbml.Parameter`
    '''
    return getInfoFromList(getParameterInfo, parameters)

def getParameterInfo(parameter, returnDict=True):
    '''
    Return information from a :class:`libsbml.Parameter` object
    
    Parameters
    ----------
    parameter: :class:`libsbml.Parameter`
    
    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''
    assert isinstance(parameter, Parameter), "Parameter object expected"
    
    # capital because id is a function
    ID = parameter.getId()
    name = parameter.getName() # optional
    
    if parameter.isSetValue():
        value = parameter.getValue() # optional
    else:
        value = None
        
    if parameter.isSetUnits():
        unit = parameter.getUnits() # optional
    else:
        unit = None

    constant = parameter.getConstant()
    
    if len(name) == 0:
        name = None
    
    if returnDict:
        return {'id':ID, 'name':name, 'value':value, 'unit':unit, 'constant':constant}
    else:
        return ID, name, value, unit, constant

def getLocalParameterInfo(parameter):
    '''
    Local parameters are always constant
    '''
    ID, name, value, unit, _constant = getParameterInfo(parameter)
    return ID, name, value, unit, True