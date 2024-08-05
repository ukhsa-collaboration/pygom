'''
Created on 14 Jan 2019

@author: thomas.finnie
@author: edwin.tye
'''
import numpy as np
import sympy

from pygom.model._model_errors import InputError
from .checks_and_conversions import check_array_type

def plot_det(solution, t, stateList=None, y=None, yStateList=None):
    '''
    Plot the results of the integration

    Parameters
    ==========
    solution: :class:`numpy.ndarray`
        solution from the integration
    t: array like
        the vector of time where the integration output correspond to
    stateList: list
        name of the states, if available

    Notes
     -----
    If we have 5 states or more, it will always be arrange such
    that it has 3 columns.
    '''

    import matplotlib.pyplot

    assert isinstance(solution, np.ndarray), "Expecting an np.ndarray"
    # if not isinstance(solution, np.ndarray):
    #     raise InputError("Expecting an np.ndarray")

    # tests on solution
    if len(solution) == solution.size:
        numState = 1
    else:
        numState = len(solution[0, :])

    assert len(solution) == len(t), "Number of solution not equal to t"
    # if len(solution) != len(t):
    #     raise InputError("Number of solution not equal to t")

    if stateList is not None:
        if len(stateList) != numState:
            raise InputError("Number of state (string) should be equal " +
                             "to number of output")
        stateList = [str(i) for i in stateList]

    # tests for y
    if y is not None:
        y = check_array_type(y)
        # if type(y) != np.ndarray:
        #     y = np.array(y)

        numTargetSol = len(y)
        # we test the validity of the input first
        if numTargetSol != len(t):
            raise InputError("Number of realization of y not equal to t")
        # then obtain the information
        if y.size == numTargetSol:
            numTargetState = 1
            y = y.reshape((numTargetSol, 1))
        else:
            numTargetState = y.shape[1]

        if yStateList is None:
            if numTargetState != numState:
                if stateList is None:
                    raise InputError("Unable to identify which observations" +
                                     " the states belong to")
                else:
                    nonAuto = False
                    for i in stateList:
                        # we are assuming here that we always name our
                        # time state as \tau when it is a non-autonomous system
                        if str(i) == 'tau':
                            nonAuto = True

                    if nonAuto == True:
                        if y.shape[1] != (solution.shape[1] - 1):
                            raise InputError("Size of y not equal to yhat")
                        else:
                            yStateList = list()
                            # we assume that our observation y follows the same
                            # sequence as the states and copy over without the
                            # time component
                            for i in stateList:
                                # test
                                if str(i) != 'tau':
                                    yStateList.append(str(i))
                    else:
                        raise InputError("Size of y not equal to yhat")
            else:
                yStateList = stateList
        else:
            if numTargetState == 1:
                if yStateList in (tuple, list):
                    if len(yStateList) != numTargetState:
                        raise InputError("Number of target state not equal to y")
                    else:
                        yStateList = [str(i) for i in yStateList]
                else:
                    if isinstance(yStateList, str):
                        yStateList = [yStateList]
                    elif isinstance(yStateList, sympy.Symbol):
                        yStateList = [str(yStateList)]
                    elif isinstance(yStateList, list):
                        assert len(yStateList) == 1, "Only have one target state"
                    else:
                        raise InputError("Not recognized input for yStateList")
            else:
                if numTargetState > numState:
                    raise InputError("Number of target state cannot be larger"
                                    + " than the number of state")

    # # let's take a moment and appreciate that we have finished checking

    # note that we can probably reduce the codes here significantly but
    # i have not thought of a good way of doing it yet.
    if numState > 9:
        numFigure = int(np.ceil(numState/9.0))
        k = 0
        last = False
        # loop over all the figures minus 1
        for z in range(numFigure - 1):
            f, axarr = matplotlib.pyplot.subplots(3, 3)
            for i in range(3):
                for j in range(3):
                    axarr[i, j].plot(t, solution[:, k])
                    if stateList is not None:
                        axarr[i, j].set_title(stateList[k])
                        if yStateList is not None:
                            if stateList[k] in yStateList:
                                idx = yStateList.index(stateList[k])
                                axarr[i, j].plot(t, y[:, idx], 'r')
                        axarr[i, j].set_xlabel('Time')
                    k += 1
            # a single plot finished, now we move on to the next one

        # now we are getting to the last one
        row = int(np.ceil((numState - (9*(numFigure - 1)))/3.0))
        f, axarr = matplotlib.pyplot.subplots(row, 3)
        if row == 1:
            for j in range(3):
                if last == True:
                    break
                axarr[j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            idx = yStateList.index(stateList[k])
                            axarr[j].plot(t, y[:,idx], 'r')
                    axarr[j].set_xlabel('Time')
                    axarr[j].set_xlim([min(t), max(t)])
                k += 1
                if k == numState:
                    last = True
        else:
            for i in range(row):
                if last == True:
                    break
                for j in range(3):
                    if last == True:
                        break
                    axarr[i, j].plot(t, solution[:, k])
                    if stateList is not None:
                        axarr[i, j].set_title(stateList[k])
                        if yStateList is not None:
                            if stateList[k] in yStateList:
                                idx = yStateList.index(stateList[k])
                                axarr[i, j].plot(t, y[:,idx], 'r')
                        axarr[i, j].set_xlabel('Time')
                        axarr[i, j].set_xlim([min(t), max(t)])
                    k += 1
                    if k == numState:
                        last = True

    elif numState <= 3:
        if numState == 1:
            # we only have one state, easy stuff
            f, axarr = matplotlib.pyplot.subplots(1, 1)
            matplotlib.pyplot.plot(t, solution)
            if stateList is not None:
                matplotlib.pyplot.plot(stateList[0])
        else:
            # we can deal with it in a single plot, in the format of 1x3
            f, axarr = matplotlib.pyplot.subplots(1, numState)
            for i in range(numState):
                axarr[i].plot(t, solution[:, i])
                if stateList is not None:
                    axarr[i].set_title(stateList[i])
                    if yStateList is not None:
                        if stateList[i] in yStateList:
                            idx = yStateList.index(stateList[i])
                            axarr[i].plot(t, y[:,idx], 'r')
                    # label :)
                    axarr[i].set_xlabel('Time')

    elif numState == 4:
        # we have a total of 4 plots, nice and easy display of a 2x2.
        # Going across first before going down
        f, axarr = matplotlib.pyplot.subplots(2, 2)
        k = 0
        for i in range(2):
            for j in range(2):
                axarr[i, j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[i, j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            idx = yStateList.index(stateList[k])
                            axarr[i, j].plot(t, y[:,idx], 'r')
                    # label :)
                    axarr[i, j].set_xlabel('Time')
                k += 1
                if numState == k:
                    break
    else:
        row = int(np.ceil(numState/3.0))
        # print(row)
        f, axarr = matplotlib.pyplot.subplots(row, 3)
        k = 0
        for i in range(row):
            for j in range(3):
                axarr[i, j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[i, j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            idx = yStateList.index(stateList[k])
                            axarr[i, j].plot(t, y[:,idx], 'r')
                    axarr[i, j].set_xlabel('Time')
                k += 1
                if numState == k:
                    break
    # finish all options, now we have plotted.
    # tidy up the output.  Without tight_layout() we will have
    # numbers in the axis overlapping each other (potentially)
    f.tight_layout()
    matplotlib.pyplot.show()
