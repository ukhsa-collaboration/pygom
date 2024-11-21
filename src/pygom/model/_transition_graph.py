import re

from .base_ode_model import BaseOdeModel
from .transition import TransitionType
from sympy.matrices import MatrixBase
from ._ode_composition import getMatchingExpressionVector, _hasExpression
import matplotlib.pyplot as plt
import sympy
from ._model_verification import checkEquation

import seaborn as sns

# Functions to produce transition graph

greekLetter = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
               'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
               'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega')


def generateTransitionGraph(ode_model, file_name=None):
    """
    Generates the transition graph in graphviz given an ode model with transitions

    Parameters
    ----------
    ode_model: OperateOdeModel
        an ode model object
    file_name: str
        location of the file, if none entered, then the default directory is used

    Returns
    -------
    dot: graphviz object
    """
    assert isinstance(ode_model, BaseOdeModel), "An ode model object required"

    from graphviz import Digraph

    if file_name is None:
        dot = Digraph(comment='ode model')
    else:
        dot = Digraph(comment='ode model', filename=file_name)

    dot.body.extend(['rankdir=LR'])

    param = [str(p) for p in ode_model.param_list]
    states = [str(s) for s in ode_model.state_list]

    for s in states:
        dot.node(s)

    events=ode_model.event_list

    # get colors, TODO: limited to 9 colours, also experiment with linetype to assist
    #                   with colourblind users
    #                   pallette from: https://gist.github.com/thriveth/8560036

    cols = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    i=0
    for e_number, event in enumerate(events):
        
        col="black"
        linetype="solid"

        if len(event.transition_list)>1:
            col=cols[i]
            i+=1

        rate=_makeEquationPretty(event.rate, param)
        for transition in event.transition_list:
            mag_lab=""
            if transition._magnitude!='1':
                mag_lab=_makeEquationPretty(transition._magnitude, param)

            if transition.transition_type==TransitionType.B:
                destination=transition.destination
                s1=destination
                n0="birth"+s1+str(e_number)
                dot.node(n0, label="", shape="none", height="0", width="0")
                dot.edge(n0, s1, label=rate, headlabel=mag_lab, color=col, style=linetype)

            elif transition.transition_type==TransitionType.D:
                origin=transition.origin
                s1=origin
                n0="death"+s1+str(e_number)
                dot.node(n0, label=mag_lab, shape="none", height="0", width="0")
                dot.edge(s1, n0, label=rate, headlabel=mag_lab, color=col, style=linetype)

            elif transition.transition_type==TransitionType.T:
                origin=transition.origin
                destination=transition.destination
                s1=origin
                s2=destination

                dot.edge(s1, s2, label=rate, headlabel=mag_lab, color=col, style=linetype)
            else:
                pass

    for ode in ode_model.ode_list:
        col="black"
        linetype="dashed"

        origin=ode.origin
        s1=origin

        n0="ode"+s1

        equation=ode.equation
        
        dot.node(n0, label="", shape="none", height="0", width="0")

        # # If there is one term and it's negative, make it a death process.
        # if equation[0]=="-" and len(sympy.Add.make_args(sympy.sympify(equation)))==1:
        #     equation=equation[1:]
        #     dot.edge(s1, n0, label=equation, color=col, style=linetype)
        # else:
        #     dot.edge(n0, s1, label=equation, color=col, style=linetype)
        dot.edge(n0, s1, label=equation, color=col, style=linetype)

    return dot



def _makeEquationPretty(eq, param):
    """
    Make the equation suitable for graphviz format by converting
    beta to &beta;  and remove all the multiplication sign

    We do not process ** and convert it to a superscript because
    it is only possible with svg (which is a real pain to convert
    back to png) and only available from graphviz versions after
    14 Oct 2011
    """
    for p in param:
        if p.lower() in greekLetter:
            eq = re.sub('(\\W?)(' + p + ')(\\W?)', '\\1&' + p + ';\\3', eq)
    # eq = re.sub('\*{1}[^\*]', '', eq)
    # eq = re.sub('([^\*]?)\*([^\*]?)', '\\1 \\2', eq)
    # eq += " blah<SUP>Yo</SUP> + ha<SUB>Boo</SUB>"
    return eq


def generateDirectedDependencyGraph(ode_matrix, transition=None):
    """
    Returns a binary matrix that contains the direction of the transition in
    a state

    Parameters
    ----------
    ode_matrix: :class:`sympy.matrcies.MatrixBase`
        A matrix of size [number of states x 1].  Obtained by
        invoking :meth:`DeterministicOde.get_ode_eqn`
    transition: list, optional
        list of transitions.  Can be generated by
        :func:`getMatchingExpressionVector`

    Returns
    -------
    G: :class:`numpy.ndarray`
        Two dimensional array of size [number of state x number of transitions]
        where each column has two entry,
        -1 and 1 to indicate the direction of the transition and the state.
        All column sum to one, i.e. transition must have a source and target.
    """
    assert isinstance(ode_matrix, MatrixBase), \
        "Expecting a vector of expressions"

    if transition is None:
        transition = getMatchingExpressionVector(ode_matrix, True)
    else:
        assert isinstance(transition, list), "Require a list of transitions"

    B = np.zeros((len(ode_matrix), len(transition)))
    for i, a in enumerate(ode_matrix):
        for j, transitionTuple in enumerate(transition):
            t1, t2 = transitionTuple
            if _hasExpression(a, t1):
                B[i, j] += -1  # going out
            if _hasExpression(a, t2):
                B[i, j] += 1   # coming in
    return B
