'''
Created on 14 Jan 2019

@author: edwin.tye
'''
import numpy as np



def check_array_type(x,accept_booleans=False):
    '''
    Check to see if the type of input is suitable.  Only operate on one
    or two dimension arrays

    Parameters
    ----------
    x: array like
        which can be either a :class:`numpy.ndarray` or list or tuple
    accept_boolean: boolean
        If true boolean elements are accepted, else they are not.

    Returns
    -------
    x: :class:`numpy.ndarray`
        checked and converted array
    '''
    accepted_types = (int,float,complex)
    if accept_booleans==True:
        type_error_message = 'Expecting elements/sub-elements to be of type float, int, complex or boolean'
    if accept_booleans==False:
        type_error_message = 'Expecting elements/sub-elements to be of type float, int or complex'
        
        
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, (list, tuple)):
        if all(isinstance(item, accepted_types) for item in x):
            if accept_booleans==True:
                x = np.array(x)
            elif accept_booleans==False:
                if  any(isinstance(item, bool) for item in x):
                    raise TypeError('No elements of array type object should be Boolean values')
                x = np.array(x)
            else:
                TypeError(type_error_message)
        elif isinstance(x[0], (list, tuple, np.ndarray)):
            for item in x:
                if any(not isinstance(sub_item, accepted_types) for sub_item in item):
                    raise TypeError(type_error_message)
                if accept_booleans==False and any(isinstance(sub_item, bool) for sub_item in item):
                    raise TypeError('No elements of array type object should be Boolean values')
            x = np.array(x)
        else:
            raise TypeError(type_error_message + ' got ' + str(type(x)))
    elif isinstance(x, accepted_types):
        if accept_booleans==True:
            x = np.array([x])
        elif accept_booleans==False and not isinstance(x, bool):
            x = np.array([x])
        else:
            TypeError("Not expecting Boolean value")
    else:
        raise TypeError("Expecting an array like object, got %s" % type(x))

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
        raise AssertionError("The number of observations and time points " +
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
        raise TypeError("Expecting a string or list")


def none_or_empty_list(x):
    y = False
    if x is not None:
        if hasattr(x, '__iter__'):
            if len(x) == 0:
                y = True
    else:
        y = True

    return y
