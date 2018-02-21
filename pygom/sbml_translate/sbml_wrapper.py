import re

from libsbml import SBMLReader
from pygom import Transition, DeterministicOde
from ._compartments import getCompartmentsInfo
from ._species import getSpeciesInfo
from ._reactions import getReactionsInfo
from ._model import getModelComponents # , getModelInfo

def readModelFromFile(filePath):
    reader = SBMLReader()
    document = reader.readSBML(filePath)

    assert document.getNumErrors() == 0, "Error detected in sbml file"

    model = document.getModel()
    return(model)

def getOdeObject(model):
    a = getModelComponents(model)
    #paramList = map(lambda x: x['id'], getCompartmentsInfo(a['comps']))
    
    param_eval = map(lambda x: (x['id'], x['size']), getCompartmentsInfo(a['comps']))
    stateList = map(lambda x: x['id'], getSpeciesInfo(a['species']))
    x0 = map(lambda x: x['x0'], getSpeciesInfo(a['species']))

    # origList = list()
    # destList = list()
    # eqnList = list()
    transition = list()
    for r in getReactionsInfo(a['reacts']):
        orig = [reactant['specie'] for reactant in r['reactant']]
        dest = [product['specie'] for product in r['product']]
        eqn = r['kineticlaw']['eqn']
        # eqnList.append(eqn)
        # paramList += map(lambda x: x['id'], r['kineticlaw']['parameters'])
        paramLocal = map(lambda x: (x['id'], x['value']), r['kineticlaw']['parameters'])

        # newTerm = map(lambda x: r['id'] + '_' + x[0], paramLocal)
        # term = map(lambda x: r'\b%s\b' % x[0], paramLocal)
        # this the first line below essentially create the two variables
        # above on the fly 
        for term in map(lambda x: x[0], paramLocal):
            eqn = re.sub(r'\b%s\b' % term, ' %s_%s ' % (r['id'], term), eqn)
        
        transition.append(Transition(orig, eqn, 'T', dest, r['id']))

        param_eval += map(lambda x: (r['id'] + '_' + x[0], x[1]), paramLocal)
#         print "\n"
#         print eqn
#         print paramLocal

    paramList = map(lambda x: x[0], param_eval)

#     print "\nfinal param_eval"+str(param_eval)
#     print paramList
#     print transition 

    ode = DeterministicOde(stateList, paramList, transition=transition)
    ode = ode.initial_values(x0).setParameters(param_eval)
    return(ode)
