from libsbml import UnitKind_toString

from ._generic import getInfoFromList

base_unit = set(
    ['ampere', 'avogadro', 'becquerel', 'candela', 'coulomb', 'dimensionless',
     'farad', 'gram', 'gray', 'henry', 'hertz', 'item',
     'joule', 'katal', 'kelvin', 'kilogram', 'litre', 'lumen',
     'lux', 'metre', 'mole', 'newton', 'ohm', 'pascal',
     'radian', 'second', 'siemens', 'sievert', 'steradian', 'tesla',
     'volt', 'watt', 'weber']
    )

def getUnitDefinitionsInfo(unitDefs):
    '''
    Return information from a list of :class:`libsbml.UnitDefinition` objects

    Parameters
    ----------
    Parameters
    ----------
    unitDefs: iterable
        an iterable where elements are :class:`libsbml.UnitDefinition`
    '''
    return getInfoFromList(getUnitDefinitionInfo, unitDefs)

def getUnitDefinitionInfo(unitDef):
    '''
    Return information from a :class:`libsbml.UnitDefinition` object

    Parameters
    ----------
    unitDef: :class:`libsbml.UnitDefinition`
    '''
    ID = unitDef.getId()
    name = unitDef.getName()
    
    listUnits = getUnitsInfo(unitDef.getListOfUnits())
    return ID, name, listUnits

def getUnitsInfo(units):
    '''
    Return information from a list of :class:`libsbml.Unit` objects

    Parameters
    ----------
    units: iterable
        an iterable where elements are :class:`libsbml.Unit`
    '''
    return getInfoFromList(getUnitInfo, units)

def getUnitInfo(unit):
    '''
    Return information from a :class:`libsbml.Unit` object

    Parameters
    ----------
    unit: :class:`libsbml.Unit`
    '''
    kind = unit.getKind()
    if isinstance(kind, (int, float)):
        kind = UnitKind_toString(kind)
    
    assert kind in base_unit, "Base unit not recognized: %s" % kind
    #kind = kind if kind in base_unit else None

    # u_new = (multiplier 10^{scale} u_kind)^{exponent}
    return unit.getExponentAsDouble(), unit.getScale(), unit.getMultiplier(), kind
