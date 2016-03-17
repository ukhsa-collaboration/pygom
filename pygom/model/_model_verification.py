import numpy

# Although reimporting * is not generally recommended
# we have to do it here so that it has all the mathematical
# functions ready to use when evaluating the equations.
# An alternative is to check for all the maths functions, such
# as exp, log, trigonometric etc.. and convert them to sympy
#from sympy.functions import (tan, cos, sin,asin, acos, atan, atan2, acot, cot, sec, csc)
from sympy.functions.elementary.trigonometric import (tan, cos, sin,
    asin, acos, atan, atan2, acot, cot, sec, csc)
from sympy.functions.elementary.exponential import (exp_polar, exp, log,
    LambertW)
from sympy.functions.combinatorial.factorials import (factorial, factorial2,
    rf, ff, binomial, RisingFactorial, FallingFactorial, subfactorial)
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, root, Min, Max,
    Id, real_root, cbrt)
from sympy.core.numbers import pi
from sympy import simplify, symbols, Expr
# from sympy.physics.units import avogadro, mol

def simplifyEquation(inputStr):
    '''
    Only simplify the equation if there is no obvious problem
    Equation is not simplified if it includes the following terms:
        exp
        log
    '''
    sList = list()
    # these are considered the "dangerous" operation that will
    # overflow/underflow in numpy
    sList.append(len(inputStr.atoms(exp)))
    sList.append(len(inputStr.atoms(log)))

    if numpy.sum(sList) != 0:
        # it is dangerous to simplify!
        return inputStr, True
    else:
        return simplify(inputStr), False
    
def checkEquation(_inputStr, _listOfVariablesDict, _derivedVariableDict):
    '''
    Convert a string into an equation and checks its validity
    '''
    if hasattr(_inputStr, '__iter__'):
        # functional programming approach
        return [checkEquation(_s, _listOfVariablesDict) for _s in _inputStr]
    else:
        assert isinstance(_inputStr, str), "Equation should be in string format"
        # create the symbols in the local environment
        for _d in _listOfVariablesDict:
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
        for _key, _value in _derivedVariableDict.iteritems():
            _isReal = True if _value.is_real else False
            exec("""%s = symbols('%s', real=%s)""" % (_key, _key, _isReal))
        # if the evaluation fails then there is a problem with the
        # variables (either state or parameters), success means that
        # it returns a symbolic expression 
        _eqn = eval(_inputStr)
        # print _inputStr, type(_eqn), isinstance(_eqn, Expr)
        # because these are the derived parameters, we need to substitute
        # them back in the formula
        if isinstance(_eqn, Expr):
            for _key, _value in _derivedVariableDict.iteritems():
                # print "\n","_eqn.subs(%s, %s)" % (_key, _value)
                _eqn = eval("_eqn.subs(%s, %s)" % (_key, _value))
        return _eqn 