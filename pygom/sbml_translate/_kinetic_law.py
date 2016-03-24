from libsbml import formulaToString, KineticLaw
from pygom.sbml_translate._parameters import getParameterInfo

def getKineticLawInfo(kineticLaw, returnDict=True):
    '''
    Return information of :class:`libsbml.KineticLaw` object
    
    Parameters
    ----------
    kineticLaw: :class:`libsbml.KineticLaw`

    returnDict: bool, optional
        whether information should be returned as a dictionary
    '''
    assert isinstance(kineticLaw, KineticLaw), "KineticLaw object expected"
    
    listParameters1 = map(getParameterInfo, kineticLaw.getListOfParameters())
    listParameters2 = map(getParameterInfo, kineticLaw.getListOfLocalParameters())

#     should we remove duplicated entry? and how
    parameters = listParameters1 + listParameters2
    eqn = formulaToString(kineticLaw.getMath())

    if returnDict:
        return {'eqn':eqn, 'parameters':parameters}
    else:
        return eqn, parameters