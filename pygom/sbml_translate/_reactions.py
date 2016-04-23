from ._kinetic_law import getKineticLawInfo
from ._species import getSpecieReferencesInfo
from ._generic import getInfoFromList

def getReactionsInfo(reactions):
    '''
    Return information from a list of :class:`libsbml.Reaction` objects

    Parameters
    ----------
    reactions: iterable
        an iterable with elements :class:`libsbml.Reaction`
    '''
    return getInfoFromList(getReactionInfo, reactions)

def getReactionInfo(reaction, returnDict=True):
    '''
    Returns information from a :class:`libsbml.Reaction` objection

    Parameters
    ----------
    reaction: :class:`libsbml.Reaction`

    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''
    ID = reaction.getId()
    name = reaction.getName() # optional
    name = name if len(name) != 0 else None

    reversible = reaction.getReversible()
    fast = reaction.getFast()

    compartment = reaction.getCompartment() # optional
    compartment = compartment if len(compartment) != 0 else None

    # although this might be an empty list, we have dealt with that
    # within the function :func:`getSpecieReferencesInfo`
    react = getSpecieReferencesInfo(reaction.getListOfReactants()) 
    prod = getSpecieReferencesInfo(reaction.getListOfProducts())

    # TODO:
    # reaction.getListOfModifiers()

    kineticLaw = getKineticLawInfo(reaction.getKineticLaw())
    if returnDict:
        return {'id':ID, 'name':name, 'reversible':reversible, 'fast':fast, 'comp':compartment, 'reactant':react, 'product':prod, 'kineticlaw':kineticLaw}
    else:
        return ID, name, reversible, fast, compartment, react, prod, kineticLaw