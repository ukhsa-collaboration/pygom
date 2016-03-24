def getInfoFromList(func, obj):
    '''
    Returns func(obj).  If obj is not an iterable, we convert it
    to a list to perform the :func:`map` of it
    
    Parameters
    ----------
    func: callable
        function to apply
    obj: object or iterable
        object as the input argument to func
    '''
    assert hasattr(func, '__call__'), "func is not a callable"

    if obj is None:
        return None
    elif hasattr(obj, '__iter__'):
        if len(obj) == 0:
            return None
        else:
            return map(func, obj)
    else:
        return map(func, [obj])