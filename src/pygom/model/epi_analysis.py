"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module containing functions that performs epidemiology based analysis
    via algebraic manipulation, such as the basic reproduction number

"""

import sympy

from .simulate import SimulateOde

__all__ = [
           'DFE',
           'R0',
           'R0_from_matrix',
           'disease_progression_matrices'
           ]

def DFE(ode, disease_state):
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

    eqn = ode.get_ode_eqn()
    index = ode.get_state_index(disease_state)
    states = [s for s in ode._iterStateList()]
    states_subs = {states[i]: 0 for i in index}
    eqn = eqn.subs(states_subs)

    DFE_solution = sympy.solve(eqn, states)
    if len(DFE_solution) == 0: DFE_solution = {}

    for s in states:
        if s not in states_subs.keys() and s not in DFE_solution.keys():
            DFE_solution.setdefault(s, 0)
    return DFE_solution

def R0(ode, disease_state):
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

    F, V = disease_progression_matrices(ode, disease_state)
    ## index = ode.get_state_index(disease_state)
    e = R0_from_matrix(F, V)
    DFE_eqn = DFE(ode, disease_state)
    e = [eig.subs(DFE_eqn) for eig in e]
    if ode.parameters is not None:
        e = [eig.subs(ode.parameters) for eig in e]

    e = list(filter(lambda x: sympy.Integer(-1) not in x.args, e))
    return (e if len(e) > 1 else e[0])

def R0_from_matrix(F, V, disease_state=None):
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
    disease_state: list like, optional
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

    if disease_state is None:
        dF = F
        dV = V
    else:
        dF = F.jacobian(disease_state)
        dV = F.jacobian(disease_state)

    K = dF*dV.inv()
    e = K.eigenvals().keys()
    e = filter(lambda x: x != 0, e)
    return list(e)

def disease_progression_matrices(ode, disease_state, diff=True):
    '''
    Returns (F,V), the secondary infection rates and disease progression
    rate respectively.

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
        The progression matrices.  If diff=False, then we return the
        :math:`F_{i}` and :math:`V_{i}` matrices as per [Brauer2008]_.
    '''

    diseaseIndex = ode.get_state_index(disease_state)
    state_list = list()
    for i, s in enumerate(ode._iterStateList()):
        if i in diseaseIndex:
            state_list.append(s)

    FList = list()
    for t in ode.transition_list:
        orig = _get_single_state_name(t.origin)
        dest = _get_single_state_name(t.destination)
        if isinstance(orig, str) and isinstance(dest, str):
            if orig not in disease_state and dest in disease_state:
                FList.append(t)

    ode2 = SimulateOde(ode.state_list, ode.param_list, transition=FList)

    F = ode2.get_ode_eqn().row(diseaseIndex)
    V = F - ode.get_ode_eqn().row(diseaseIndex)

    if diff:
        dF = F.jacobian(state_list)
        dV = V.jacobian(state_list)
        return dF, dV
    else:
        return F,V


def _get_single_state_name(state):
    if hasattr(state, '__iter__'):
        if isinstance(state, str):
            return state
        else:
            state = state[0] if len(state) == 1 else None
    if isinstance(state, str):
        return state
    elif isinstance(state, sympy.Symbol):
        return str(state)
    else:
        return None
