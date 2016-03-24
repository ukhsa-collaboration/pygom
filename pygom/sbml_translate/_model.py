def getModelInfo(model, returnDict=True):
    '''
    Return information from an :class:`libsbml.Model` object.
    Only the top layer is extracted.  To get the second layer
    of information use :func:`getModelComponents`

    Parameters
    ----------
    model: :class:`libsbml.Model`
        a Model object
    returnDict: bool, optional
        whether information should be returned as a dictionary
    
    See Also
    --------
    :func:`getModelComponents`
    '''   
    ID = strOrNone(model.getId())
    name = strOrNone(model.getName())
    substance = strOrNone(model.getSubstanceUnits())
    time = strOrNone(model.getTimeUnits())
    volume = strOrNone(model.getVolumeUnits())
    area = strOrNone(model.getAreaUnits())
    length = strOrNone(model.getLengthUnits())
    extent = strOrNone(model.getExtentUnits())
    conversion = strOrNone(model.getConversionFactor())
    
    if returnDict:
        return {'id':ID, 'name':name, 'substance':substance, 'time':time,
                'volume':volume, 'area':area, 'length':length, 
                'extent':extent, 'conversion':conversion}
    else:
        return ID, name, substance, time, volume, area, length, extent, conversion

def getModelComponents(model):
    '''
    Return components from a :class:`libsbml.Model` object.
    The second level information is extracted.  To get the global model
    information use :func:`getModelInfo`

    Parameters
    ----------
    model: :class:`libsbml.Model`
    
    See Also
    --------
    :func:`getModelIfo`
    '''
    tmpDict = dict()
    # in sequence of the SBML specification
    tmpDict['funcDefs'] = model.getListOfFunctionDefinitions()
    tmpDict['unitDefs'] = model.getListOfUnitDefinitions()
    tmpDict['comps'] = model.getListOfCompartments()
    tmpDict['species'] = model.getListOfSpecies()
    tmpDict['params'] = model.getListOfParameters()
    tmpDict['assigns'] = model.getListOfInitialAssignments()
    tmpDict['rules'] = model.getListOfRules()
    tmpDict['cons'] = model.getListOfConstraints()
    tmpDict['reacts'] = model.getListOfReactions()
    tmpDict['events'] = model.getListOfEvents()

    for key in tmpDict.keys():
        if len(tmpDict[key]) == 0:
            del tmpDict[key]

    return tmpDict

def strOrNone(inputStr):
    '''
    Returns a string of length zero as none
    '''
    assert isinstance(inputStr, str), "Expecting a string input but found %s" % type(inputStr)
    if len(inputStr) != 0:
        return inputStr
    else:
        return None