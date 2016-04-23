from ._generic import getInfoFromList

def getCompartmentsInfo(compartments):
    '''
    Return information from a list of :class:`libsbml.Compartment` objects

    Parameters
    ----------
    compartments: iterable
        an iterable with elements :class:`libsbml.Compartment`
    '''
    return getInfoFromList(getCompartmentInfo, compartments)

def getCompartmentInfo(compartment, returnDict=True):
    '''
    Return information from an :class:`libsbml.Compartment` object

    Parameters
    ----------
    compartment: :class:`libsbml.Compartment`

    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''    
    ID = compartment.getId()
    name = compartment.getName() # optional
    name = name if len(name) != 0 else None
    
    if compartment.isSetSpatialDimensions():
        spatial = compartment.getSpatialDimensionsAsDouble() # optional
        if not isinstance(spatial, float):
            spatial = None
    else:
        spatial = None

    size = compartment.getSize() # optional
    if compartment.isSetSize():
        size = compartment.getSize()
    else:
        size = None
    
    if compartment.isSetUnits():
        unit = compartment.getUnits() # optional
    else:
        unit = None

    constant = compartment.getConstant()

    if returnDict:
        return {'id':ID, 'name':name , 'spatial':spatial, 'size':size, 'unit':unit, 'constant':constant}
    else:
        return ID, name , spatial, size, unit, constant