"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Functions that is used to determine the composition of the
    defined ode

"""
import re
from functools import reduce

import sympy
from sympy.matrices import MatrixBase
import numpy as np

from .base_ode_model import BaseOdeModel
from .transition import TransitionType

def getUnmatchedExpressionVector(expr_vec, full_output=False):
    """
    Return the unmatched expressions from a vector of ODE equations.
    e.g [a+b, -a+c, -c+b-d] returns [b, -d, b] 

    Parameters
    ----------
    expr_vec: :class:`sympy.matrices.MatrixBase`
        A matrix of size [number of states x 1].
    full_output: bool, optional
        Defaults to False, if True, also output the list of matched expressions

    Returns
    -------
    list:
        of unmatched expressions, i.e. birth or death processes
    """
    assert isinstance(expr_vec, MatrixBase), \
        "Expecting a vector of expressions"

    transition = reduce(lambda x, y: x + y, map(getExpressions, expr_vec))
    matched_transition_list = _findMatchingExpression(transition)
    out = list(set(transition) - set(matched_transition_list))

    if full_output:
        return out, _transitionListToMatchedTuple(matched_transition_list)
    else:
        return out


def getMatchingExpressionVector(expr_vec, outTuple=False):
    """
    Return the matched expressions from a vector of equations

    Parameters
    ----------
    expr_vec: :class:`sympy.matrices.MatrixBase`
        A matrix of size [number of states x 1].
    outTuple: bool, optional
        Defaults to False, if True, the output is a tuple of length two
        which has the matching elements.  The first element is always
        positive and the second negative

    Returns
    -------
    list:
        of matched expressions, i.e. transitions
    """
    assert isinstance(expr_vec, MatrixBase), \
        "Expecting a vector of expressions"

    transition = list()
    for expr in expr_vec:
        transition += getExpressions(expr)

    transition = list(set(_findMatchingExpression(transition)))

    if outTuple:
        return _transitionListToMatchedTuple(transition)
    else:
        return transition


def _findMatchingExpression(expressions, full_output=False):
    """
    Reduce a list of expressions to a list of transitions.  A transition
    is found when two expressions are identical with a change of sign.

    Parameters
    ----------
    expressions: list
        the list of expressions
    full_output: bool, optional
        If True, output the unmatched expressions as well. Defaults to False.

    Returns
    -------
    list:
        of expressions that was matched
    """
    t_list = list()
    for i in range(len(expressions) - 1):
        for j in range(i + 1, len(expressions)):
            b = expressions[i] + expressions[j]
            if b == 0:
                t_list.append(expressions[i])
                t_list.append(expressions[j])

    if full_output:
        unmatched = set(expressions) - set(t_list)
        return t_list, list(unmatched)
    else:
        return t_list


def _transitionListToMatchedTuple(transition):
    """
    Convert a list of transitions to a list of tuple, where each tuple
    is of length 2 and contains the matched transitions. First element
    of the tuple is the positive term
    """
    t_tuple_list = list()
    for i in range(len(transition) - 1):
        for j in range(i + 1, len(transition)):
            b = transition[i] + transition[j]
            # the two terms cancel out
            if b == 0:
                if sympy.Integer(-1) in getLeafs(transition[i]):
                    t_tuple_list.append((transition[j], transition[i]))
                else:
                    t_tuple_list.append((transition[i], transition[j]))
    return t_tuple_list


def getExpressions(expr):
    input_dict = dict()
    _getExpression(expr.expand(), input_dict)
    return list(input_dict.keys())


def getLeafs(expr):
    input_dict = dict()
    _getLeaf(expr.expand(), input_dict)
    return list(input_dict.keys())


def _getLeaf(expr, input_dict):
    """
    Get the leafs of an expression, can probably just do
    the same with expr.atoms() with most expression but we
    do not break down power terms i.e. x**2 will be broken
    down to (x,2) in expr.atoms() but this function will
    retain (x**2)
    """
    t = expr.args
    t_lengths = np.array(list(map(_expressionLength, t)))

    for i, ti in enumerate(t):
        if t_lengths[i] == 0:
            input_dict.setdefault(ti, 0)
            input_dict[ti] += 1
        else:
            _getLeaf(ti, input_dict)


def _getExpression(expr, input_dict):
    """
    all the operations is dependent on the conditions 
    whether all the elements are leafs or only some of them.
    Only return expressions and not the individual elements
    """
    t = expr.args if len(expr.atoms()) > 1 else [expr]
    # print t

    # find out the length of the components within this node
    t_lengths = np.array(list(map(_expressionLength, t)))
    # print(tLengths)
    if np.all(t_lengths == 0):
        # if all components are leafs, then the node is an expression
        input_dict.setdefault(expr, 0)
        input_dict[expr] += 1
    else:
        for i, ti in enumerate(t):
            # if the leaf is a singleton, then it is an expression
            # else, go further along the tree
            if t_lengths[i] == 0:
                input_dict.setdefault(ti, 0)
                input_dict[ti] += 1
            else:
                if isinstance(ti, sympy.Mul):
                    _getExpression(ti, input_dict)
                elif isinstance(ti, sympy.Pow):
                    input_dict.setdefault(ti, 0)
                    input_dict[ti] += 1


def _expressionLength(expr):
    """
    Returns the length of the expression i.e. number of terms.
    If the expression is a power term, i.e. x^2 then we assume
    that it is one term and return 0.
    """
    # print type(expr)
    if isinstance(expr, sympy.Mul):
        return len(expr.args)
    elif isinstance(expr, sympy.Pow):
        return 0
    else:
        return 0


def _findIndex(eq_vec, expr):
    """
    Given a vector of expressions, find where you will locate the
    input term.

    Parameters
    ----------
    eq_vec: :class:`sympy.Matrix`
        vector of sympy equation
    expr: sympy type
        An expression that we would like to find

    Returns
    -------
    list:
        of index that contains the expression.  Can be an empty list
        or with multiple integer
    """
    out = list()
    for i, a in enumerate(eq_vec):
        j = _hasExpression(a, expr)
        if j is True:
            out.append(i)
    return out


def _hasExpression(eq, expr):
    """
    Test whether the equation eq has the expression expr
    """
    out = False
    aExpand = eq.expand()
    if expr == aExpand:
        out = True
    if expr in aExpand.args:
        out = True
    return out


def pureTransitionToOde(A):
    """
    Get the ode from a pure transition matrix

    Parameters
    ----------
    A: `sympy.Matrix`
        a transition matrix of size [n \times n]

    Returns
    -------
    b: `sympy.Matrix`
        a matrix of size [n \times 1] which is the ode
    """
    nrow, ncol = A.shape
    assert nrow == ncol, "Need a square matrix"
    B = [sum(A[:, i]) - sum(A[i, :]) for i in range(nrow)]
    return sympy.simplify(sympy.Matrix(B))


def stripBDFromOde(fx, bd_list=None):
    if bd_list is None:
        bd_list = getUnmatchedExpressionVector(fx, False)

    fx_copy = fx.copy()
    for i, fxi in enumerate(fx):
        term_in_expr = list(map(lambda x: x in fxi.expand().args, bd_list))
        for j, term in enumerate(bd_list):
            fx_copy[i] -= term if term_in_expr[j] else 0
    # simplify converts it to an ImmutableMatrix, so we make it into
    # a mutable object again because we want the expanded form

    # simplify() causes issues when we have terms with denominators as
    # it will try to give all terms a common denominator thus 
    # potentially masking terms which should be matched.
    # We try leaving it out for now, but some thorough testing is required
    # of the unroll pipeline.

    # return sympy.Matrix(sympy.simplify(fx_copy)).expand()
    return sympy.Matrix(fx_copy).expand()


def odeToPureTransition(fx, states, output_remain=False):
    bd_list, term_list = getUnmatchedExpressionVector(fx, full_output=True)
    fx = stripBDFromOde(fx, bd_list)
    # we now have fx with pure transitions
    A, remain_terms = _singleOriginTransition(fx, term_list, states)
    A, remain_terms = _odeToPureTransition(fx, remain_terms, A)
    # checking if our decomposition is correct
    fx1 = pureTransitionToOde(A)
    diff_ode = sympy.simplify(fx - fx1)
    if np.all(np.array(map(lambda x: x == 0, diff_ode)) == True):
        if output_remain:
            return A, remain_terms
        else:
            return A
    else:
        diff_term = sympy.Matrix(list(filter(lambda x: x != 0, diff_ode)))
        diff_term_list = getMatchingExpressionVector(diff_term, True)
        # If there is some single origin transition not being matched up
        # it is most likely because the transition originates from a
        # combination like (1-x) which got split into two parts - the
        # "1" and the "x" part.  So we try to reverse the sign to see
        # if it helps.
        # TODO: increase robustness so if it does not help, then we
        # either bail out or revert to the normal version
        diff_term_list = map(lambda x_y: (x_y[1], x_y[0]), diff_term_list)
        A, remain_terms = _singleOriginTransition(diff_ode, diff_term_list,
                                                  states, A)
        AA, remain_terms = _odeToPureTransition(diff_ode, remain_terms, A)

        if output_remain:
            return AA, remain_terms
        else:
            return AA


def _odeToPureTransition(fx, terms=None, A=None):
    """
    Get the pure transition matrix between states

    Parameters
    ----------
    fx: :class:`sympy.matrices.MatrixBase`
       input ode with pure transitions in symbolic form, :math:`f(x)`
    terms:
        list of two element tuples which contains the
        matching terms
    A:  `sympy.matricies.MatrixBase`, optional
        the matrix to be filled.  Defaults to None, which
        will lead to the creation of a [len(fx), len(fx)] matrix
        with all zero elements
    Returns
    -------
    A: :class:`sympy.matricies.MatrixBase`
        resulting transition matrix
    remain: list
        list of  which contains the unmatched
        transitions
    """
    if terms is None:
        terms = getMatchingExpressionVector(fx, True)

    if A is None:
        A = sympy.zeros(len(fx), len(fx))

    remain_transition = list()
    for t1, t2 in terms:
        remain = True
        for i, aFrom in enumerate(fx):
            if _hasExpression(aFrom, t2):
                # arriving at
                for j, aTo in enumerate(fx):
                    if _hasExpression(aTo, t1):
                        A[i, j] += t1  # from i to j
                        remain = False
        if remain:
            remain_transition.append((t1, t2))

    return A, remain_transition


def _singleOriginTransition(fx, term_list, states, A=None):
    if A is None:
        A = sympy.zeros(len(fx), len(fx))

    remain_term_list = list()
    for k, transition_tuple in enumerate(term_list):
        t1, t2 = transition_tuple
        possible_origin = list()
        remain = True
        for i, s in enumerate(states):
            if s in t1.atoms():
                possible_origin.append(i)
        if len(possible_origin) == 1:
            for j, fxj in enumerate(fx):
                # print(t1, fxj, possibleOrigin[0] != j, _hasExpression(fxj, t2))
                if possible_origin[0] != j and _hasExpression(fxj, t1):
                    A[possible_origin[0], j] += t1
                    remain = False
                    # print(t1, possibleOrigin, j, fxj, "\n")
        if remain:
            remain_term_list.append(transition_tuple)

    return A, remain_term_list
