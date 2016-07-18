import re

from libsbml import SBMLReader
from pygom import Transition, OperateOdeModel
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
    
    paramEval = map(lambda x: (x['id'], x['size']), getCompartmentsInfo(a['comps']))
    stateList = map(lambda x: x['id'], getSpeciesInfo(a['species']))
    x0 = map(lambda x: x['x0'], getSpeciesInfo(a['species']))

    # origList = list()
    # destList = list()
    # eqnList = list()
    transitionList = list()
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
        
        transitionList.append(Transition(orig, eqn, 'T', dest, r['id']))

        paramEval += map(lambda x: (r['id'] + '_' + x[0], x[1]), paramLocal)
#         print "\n"
#         print eqn
#         print paramLocal

    paramList = map(lambda x: x[0], paramEval)

#     print "\nfinal paramEval"+str(paramEval)
#     print paramList
#     print transitionList 

    ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
    ode = ode.setInitialState(x0).setParameters(paramEval)
    return(ode)
