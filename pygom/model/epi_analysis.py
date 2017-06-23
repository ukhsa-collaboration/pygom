"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module containing functions that performs epidemiology based analysis
    via algebraic manipulation, such as the basic reproduction number

"""

import sympy

from .stochastic import SimulateOdeModel

__all__ = [
           'getDFE',
           'getR0',
           'getR0GivenMatrix',
           'getDiseaseProgressionMatrices'
           ]

def getDFE(ode, diseaseState):
    '''
    Returns the disease free equilibrium from an ode object

    Parameters
    ----------
    ode: :class:`.BaseOdeModel`
        a class object from pygom
    diseaseState: array like
        name of the disease states
    
    Returns
    -------
    e: array like
        disease free equilibrium

    '''

    eqn = ode.getOde()
    index = ode.getStateIndex(diseaseState)
    states = [s for s in ode._iterStateList()]
    statesSubs = {states[i]:0 for i in index}
    eqn = eqn.subs(statesSubs)

    DFE = sympy.solve(eqn, states)
    if len(DFE) == 0: DFE = {}

    for s in states:
        if s not in statesSubs.keys() and s not in DFE.keys():
            DFE.setdefault(s, 0)
    return DFE

def getR0(ode, diseaseState):
    '''
    Returns the basic reproduction number, in symbolic form when
    the parameter values are not available

    Parameters
    ----------
    ode: :class:`.BaseOdeModel`
        a class object from pygom
    diseaseStateIndex: array like
        name of the disease states
    
    Returns
    -------
    e: array like
        R0

    See Also
    --------
    :func:`getDiseaseProgressionMatrices`, :func:`getR0GivenMatrix`

    '''

    F, V = getDiseaseProgressionMatrices(ode, diseaseState)
    ## index = ode.getStateIndex(diseaseState)
    e = getR0GivenMatrix(F, V)
    DFE = getDFE(ode, diseaseState)
    e = [eig.subs(DFE) for eig in e]
    if ode.getParameters() is not None:
        e = [eig.subs(ode.getParameters()) for eig in e]

    e = list(filter(lambda x: sympy.Integer(-1) not in x.args, e))
    return (e if len(e) > 1 else e[0])

def getR0GivenMatrix(F, V, diseaseState=None):
    '''
    Returns the symbolic form of the basic reproduction number. This will
    include the states symbols which is different from :func:`getR0` where
    the states is replaced by the values of the disease-free equilibrium.

    Parameters
    ----------
    F: :class:`sympy.matrices.MatrixBase`
        secondary infection rates        
    V: :class:`sympy.matrices.MatrixBase`
        disease progression rates
    diseaseState: list like, optional
        list of the disease state as :class:`sympy.Symbol`.  Defaults
        to None which assumes that :math:`F,V` had been differentiated
    
    Returns
    -------
    e: :class:`sympy.matrices.MatrixBase`
        the eigenvalues of :math:`FV^{-1}` for the disease states

    See Also
    --------
    :func:`getDiseaseProgressionMatrices`, :func:`getR0`
    '''

    if diseaseState is None:
        dF = F
        dV = V
    else:
        dF = F.jacobian(diseaseState)
        dV = F.jacobian(diseaseState)

    K = dF*dV.inv()
    e = K.eigenvals().keys()
    e = filter(lambda x: x != 0, e)
    return list(e)

def getDiseaseProgressionMatrices(ode, diseaseState, diff=True):
    '''
    Returns (F,V), the secondary infection rates and disease progression rate
    respectively.

    Parameters
    ----------
    ode: :class:`.BaseOdeModel`
        an ode class in pygom
    diseaseStates: array like
        the name of the disease states
    diff: bool, optional
        if the first derivative of the matrices are return, defaults to true
    
    Returns
    -------
    (F, V): tuple
        The progression matrices.  If diff=False, then we return the :math:`F_{i}` and
        :math:`V_{i}` matrices as per [Brauer2008]_.
    '''

    diseaseIndex = ode.getStateIndex(diseaseState)
    stateList = list()
    for i, s in enumerate(ode._iterStateList()):
        if i in diseaseIndex:
            stateList.append(s)

    FList = list()
    for t in ode.getTransitionList():
        orig = _getSingleStateName(t.getOrigState())
        dest = _getSingleStateName(t.getDestState())
        if isinstance(orig, str) and isinstance(dest, str):
            if orig not in diseaseState and dest in diseaseState:
                FList.append(t)

    ode2 = SimulateOdeModel(ode.getStateList(), 
                            ode.getParamList(), 
                            transitionList=FList)

    F = ode2.getOde().row(diseaseIndex)
    V = F - ode.getOde().row(diseaseIndex)

    if diff:
        dF = F.jacobian(stateList)
        dV = V.jacobian(stateList)
        return dF, dV
    else:
        return F,V


def _getSingleStateName(state):
    if hasattr(state, '__iter__'):
        state = state[0] if len(state) == 1 else None
    if isinstance(state, str):
        return state
    elif isinstance(state, sympy.Symbol):
        return str(state)
    else:
        return None
