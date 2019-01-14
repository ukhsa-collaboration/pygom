'''
Created on 14 Jan 2019

@author: edwin.tye
'''
import numpy as np
from numbers import Number

from pygom.model._model_errors import InputError, ArrayError


def check_array_type(x):
    '''
    Check to see if the type of input is suitable.  Only operate on one
    or two dimension arrays

    Parameters
    ----------
    x: array like
        which can be either a :class:`numpy.ndarray` or list or tuple

    Returns
    -------
    x: :class:`numpy.ndarray`
        checked and converted array
    '''

    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, (list, tuple)):
        if isinstance(x[0], Number):
            x = np.array(x)
        elif isinstance(x[0], (list, tuple, np.ndarray)):
            if isinstance(x[0][0], Number):
                x = np.array(x)
            else:
                raise ArrayError("Expecting elements of float or int")
        else:
            raise ArrayError("Expecting elements of float or int")
    elif isinstance(x, Number):
        x = np.array([x])
    else:
        raise ArrayError("Expecting an array like object, got %s" % type(x))

    return x

def check_dimension(x, y):
    '''
    Compare the length of two array like objects.  Converting both to a numpy
    array in the process if they are not already one.

    Parameters
    ----------
    x: array like
        first array
    y: array like
        second array

    Returns
    -------
    x: :class:`numpy.array`
        checked and converted first array
    y: :class:`numpy.array`
        checked and converted second array
    '''

    y = check_array_type(y)
    x = check_array_type(x)

    if len(y) != len(x):
        raise InputError("The number of observations and time points " +
                         "should have the same length")

    return (x, y)

def is_list_like(x):
    '''
    Test whether the input is a type that behaves like a list, such
    as (list,tuple,np.ndarray)

    Parameters
    ----------
    x:
        anything

    Returns
    -------
    bool:
        True if it belongs to one of the three expected type
        (list,tuple,np.ndarray)
    '''
    return isinstance(x, (list, tuple, np.ndarray))

def str_or_list(x):
    '''
    Test to see whether input is a string or a list.  If it
    is a string, then we convert it to a list.

    Parameters
    ----------
    x:
        str or list

    Returns
    -------
    x:
        x in list form

    '''
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif isinstance(x, str):
        return [x]
    else:
        raise InputError("Expecting a string or list")


def none_or_empty_list(x):
    y = False
    if x is not None:
        if hasattr(x, '__iter__'):
            if len(x) == 0:
                y = True
    else:
        y = True

    return y
