import numpy as np

# Although reimporting * is not generally recommended
# we have to do it here so that it has all the mathematical
# functions ready to use when evaluating the equations.
# An alternative is to check for all the maths functions, such
# as exp, log, trigonometric etc.. and convert them to sympy
from sympy.functions.elementary.exponential import (exp_polar, exp, log,
    LambertW)
ln = log

from sympy.functions.elementary.trigonometric import (sin, cos, tan,
        sec, csc, cot, 
        sinc, 
        asin, acos, atan, 
        asec, acsc, acot, 
        atan2)
arcsin = asin
arccos = acos
arctan = atan
arcsec = asec
arccsc = acsc
arccot = acot

from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, 
        sech, csch, coth, asinh, acosh, atanh, acoth, asech)
arcsinh = asinh
arccosh = acosh
arctanh = atanh
arcsech = asech
# arccsch = acsch
arccoth = acoth

from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold        
from sympy.functions.combinatorial.factorials import (factorial, factorial2,
    rf, ff, binomial, RisingFactorial, FallingFactorial, subfactorial)
from sympy.functions.elementary.integers import floor, ceiling, frac
from sympy.functions.elementary.miscellaneous import (sqrt, root, Min, Max,
    Id, real_root, cbrt)
from sympy.functions.elementary.complexes import Abs
from sympy.core.numbers import pi
from sympy import simplify, symbols, Expr
# from sympy.physics.units import avogadro, mol
def power(a,b): a**b

def simplifyEquation(input_str):
    """
    Only simplify the equation if there is no obvious problem
    Equation is not simplified if it includes the following terms:
        exp
        log
    """
    s_list = list()
    # these are considered the "dangerous" operation that will
    # overflow/underflow in np
    s_list.append(len(input_str.atoms(exp)))
    s_list.append(len(input_str.atoms(log)))

    if np.sum(s_list) != 0:
        # it is dangerous to simplify!
        return input_str, True
    else:
        #TODO: Removed actual simplyify (do we need it?)
        return input_str, False


def checkEquation(input_str, input_var, derived_var, subs_derived=True):
    """
    Convert a string into an equation and checks its validity.  Everything
    here is prepended with an underscore to ensure that it does not pollute
    the local environment which is essential for the symbolic equations.
    An symbol starting with an underscore is not allowed, and should be
    checked prior to using this function
    """
    
    if isinstance(input_str, str):
        input_str = [input_str]
    assert hasattr(input_str, '__iter__'), "Expecting an iterable"
    
    list_out = list()
    for _inputStr in input_str:
        assert isinstance(_inputStr, str), "Equation should be in string format"
        # create the symbols in the local environment
        for _d in input_var:
            for _s in _d.keys():
                if isinstance(_d[_s], tuple):
                    # only the first element, as we made this as a vector
                    _isReal = True if _d[_s][0].is_real else False
                    _sString = [str(_sym) for _sym in _d[_s]]
                    _sConcat = ','
                    exec("""%s = symbols('%s',  real=%s)""" % (_s, _sConcat.join(_sString), _isReal))
                else:
                    _isReal = True if _d[_s].is_real else False
                    exec("""%s = symbols('%s', real=%s)""" % (_s, _s, _isReal))
        for _key, _value in derived_var.items():
            _isReal = True if _value.is_real else False
            exec("""%s = symbols('%s', real=%s)""" % (_key, _key, _isReal))
        # if the evaluation fails then there is a problem with the
        # variables (either state or parameters), success means that
        # it returns a symbolic expression 
        _eqn = eval(_inputStr)
        # print _inputStr, type(_eqn), isinstance(_eqn, Expr)
        if subs_derived:
            # because these are the derived parameters, we need to substitute
            # them back in the formula
            if isinstance(_eqn, Expr):
                for _key, _value in derived_var.items():
                    _eqn = eval("_eqn.subs(%s, %s)" % (_key, _value))
        list_out.append(_eqn)

    if len(list_out) == 1:
        return list_out[0]
    else:
        return list_out


def checkEquation2(input_str, list_vars):
    """
    Uses a functional programming approach
    """
    if hasattr(input_str, '__iter__'):
        from functools import partial
        f = partial(checkEquation2, _listOfVariablesStr=list_vars)
        return map(f, input_str)
    else:
        for _s in list_vars:
            exec("""%s = symbols('%s')""" % (_s, _s))
        _eqn = eval(input_str)
        return _eqn
