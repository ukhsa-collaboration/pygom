from pygom.model._model_errors import InputError
from ._generic import getInfoFromList
from libsbml import SpeciesReference, Species

def getSpeciesInfo(species):
    '''
    Return information from a list of :class:`libsbml.Species` objects

    Parameters
    ----------
    specie: iterable
        an iterable with elements :class:`libsbml.Species`
    '''
    return getInfoFromList(getSpecieInfo, species)

def getSpecieInfo(specie, returnDict=True):
    '''
    Return information of a :class:`libsbml.Species` object

    Parameters
    ----------
    specie: :class:`libsbml.Species`

    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''
    assert isinstance(specie, Species), "Species object expected"
    
    ID = specie.getId()
    name = specie.getName() # optional, output string
    name = name if len(name) != 0 else None
    
    comps = specie.getCompartment()
    
    x0 = specie.getInitialAmount() # optional
    z0 = specie.getInitialConcentration() # optional
    
    unit = specie.getSubstanceUnits() # optional, output string
    unit = unit if len(unit) != 0 else None

    isDensity = specie.getHasOnlySubstanceUnits()
    
    y = specie.getBoundaryCondition()
    constant = specie.getConstant()
    conversion = specie.getConversionFactor()
    
    if isDensity:
        if z0 == 0.0:
            raise InputError('Species was indicated to be a density')
    else:
        if x0 == 0.0:
            raise InputError('Species was indicated to be an amount')

    if returnDict:
        return {'id':ID, 'name':name, 'comp':comps, 'x0':x0, 'z0':z0, 'unit':unit, 'density':isDensity, 'y':y, 'constant':constant, 'conversion':conversion}
    else:
        return ID, name, comps, x0, z0, unit, isDensity, y, constant, conversion

def getSpecieReferencesInfo(specieRefs):
    '''
    Return information from a list of :class:`libsbml.SpeciesReference` objects

    Parameters
    ----------
    specieRef: iterable
        an iterable with elements of :class:`libsbml.SpeciesReference`
    '''
    return getInfoFromList(getSpecieReferenceInfo, specieRefs)

def getSpecieReferenceInfo(specieRef, returnDict=True):
    '''
    Return information of a :class:`libsbml.SpeciesReference` object

    Parameters
    ----------
    specieRef: :class:`libsbml.SpeciesReference`

    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''
    assert isinstance(specieRef, SpeciesReference), "SpeciesReference object expected" 
    
    ID = specieRef.getId() # optional
    ID = ID if len(ID) != 0 else None
    name = specieRef.getName() # optional
    name = name if len(name) != 0 else None
    specie = specieRef.getSpecies() # id of the species
    
    # first one is for level 3, second one for level 2 and before
    # this is also optional
    stoichiometry = specieRef.getStoichiometry() or specieRef.getStoichiometryMath()
    constant = specieRef.getConstant()
    
    if returnDict:
        return {'id':ID, 'name':name, 'specie':specie, 'stoichiometry':stoichiometry, 'constant':constant}
    else:
        return ID, name, specie, stoichiometry, constant
    