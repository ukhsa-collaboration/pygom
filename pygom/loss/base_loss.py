"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    To place everything about estimating the parameters of an ode model
    under square loss in one single module.  Focus on the standard local
    method which means obtaining the gradient and Hessian.

"""

#__all__ = [] # don't really want to export this

import copy
import scipy.integrate, scipy.interpolate, scipy.sparse, scipy.optimize
import numpy
import gc

from pygom.loss.loss_type import Square
from pygom.model import ode_utils
from pygom.model._modelErrors import InputError

class BaseLoss(object):
    '''
    This contains the base that stores all the information of an ode.

    Parameters
    ----------
    theta: array like
        input value of the parameters
    ode: :class:`OperateOdeModel`
        the ode class in this package
    x0: numeric
        initial time
    t0: numeric
        initial value
    t: array like
        time points where observations were made
    y: array like
        observations
    stateName: str
        the state which the observations came from

    '''
    def __init__(self, theta, ode,
                 x0, t0,
                 t, y,
                 stateName, stateWeight=None,
                 targetParam=None, targetState=None):

        ### Execute all the checks first

        # conversion into numpy
        t = ode_utils.checkArrayType(t)
        y = ode_utils.checkArrayType(y)
        
        if stateWeight is None:
            stateWeight = 1.0

        if len(y) == y.size:
            y = y.flatten()
            n = len(y)
            p = 1
        else:
            n, p = y.shape

        assert len(t) == n, "Number of observation and time must have equal length"
        # if len(t) == len(y):
        #     pass
        # else:
        #     raise Exception("Number of observation and time must have equal length")

        # TODO: think about whether this should be a copy
        # there are pros and cons with referencing or copy
        # if we copy, we isolate the ode so we can make a
        # distributed/parallel estimate
        # but it is easier to diagnose problems when we
        # don't copy and also make use of the plotting methods
        # because the parameters are now basically just a pointer
        # that is continuously updated
        self._ode = ode

        # We are making a shitty check here because I screwed up (sort of)
        # Should have been a base class where we do not have the targetParam and
        # targetStaet and another class extending it.  The only problem of that is the lost
        # of ability to make faster calculation, which is not even possible now because of
        # how OperateOdeModel works.  Ideally, operateOdeModel will take in the targetParam
        # in a way so that the gradient information is only computed on those targeted
        # instead of computing the full vector before extracting the relevant elements.
        # Basically, it will require a lot of work to make things sync and that is too much
        # effort and time which I do not have
        if self._ode._parameters is None:
            if len(self._ode._paramList) != 0:
                # note that this is necessary because we want to make sure that it is possible
                # to only estimate part of the full parameter set
                raise RuntimeError("Set the parameters of the ode first")
        else:
            try:
                solution = self._ode.setInitialValue(x0, t0).integrate2(t)
            except Exception as e:
                print e
                if t0 == t[1]:
                    raise InputError("First time point t[1] is equal to t0")
                else:
                    raise InputError("ode not initialized properly or unable "+
                                     "to integrate using the initial values provided")

        # Information
        self._numParam = self._ode.getNumParam()
        self._numState = self._ode.getNumState()

        ### We wish to know whether we are dealing with a multiobjective problem

        # decide whether we are working on a restricted set
        # the target parameters
        if targetParam is None:
            self._targetParam = None
        else:
            self._targetParam = ode_utils.strOrList(targetParam)

        if targetState is None:
            self._targetState = None
        else:
            self._targetState = ode_utils.strOrList(targetState)

        # check stuff
        # if you are trying to go through this, I apologize
        if stateName is None:
            # then if
            if solution.shape[1] == p:
                stateName = [str(i) for i in self._ode._stateList]
                self._setWeight(n, p, stateWeight)
            else:
                raise InputError("Expecting the name of states for the observations")
        elif isinstance(stateName, (str, list, tuple)):
            if isinstance(stateName, str):
                stateName = [stateName]

            assert p == len(stateName), "len(stateName) and len(y[0]) is not equal"
            self._setWeight(n, p, stateWeight)
        else:
            raise InputError("State name should be str or of type list/tuple")

        # if self._stateWeight is not None:
        if numpy.any(self._stateWeight <= 0):
            raise InputError("Weights should be strictly positive")

        # finish ordering information
        # now get the index of target states
        self._stateName = stateName

        # print self._stateName
        self._stateIndex = self._ode.getStateIndex(self._stateName)
        # finish

        ### now we set the scene

        # making sure that our arrays are actually arrays
        # parameters
        self._setParam(theta)
        self._setX0(x0)

        self._y = y
        self._t0 = t0

        # but the observed array t does not include the initial value
        # so we first check the type
        self._observeT = t.copy()
        # and insert the initial value
        self._t = numpy.insert(t, 0, t0)
        # and length
        self._numTime = len(self._t)

        # interpolating information
        self._interpolateTime = None
        self._interpolateTimeIndex = None

        # TODO: optimal weight in terms of Pareto front from a
        # multiobjective optimization perspective
        # print self._stateWeight
        self._lossObj = self._setLossType()

        # final check
        if self._t is None or self._y is None or self._stateName is None:
            raise InputError("Error without data currently not implemented")

    ############################################################
    #
    # Gradient operators
    #
    ############################################################

    def gradient(self,theta=None,full_output=False):
        '''
        Returns the gradient calculated by solving the forward sensitivity
        equation.  Identical to :meth:`sensitivity` without the choice of
        integration method

        See Also
        --------
        :meth:`sensitivity`
        '''
        return self.sensitivity(theta, full_output)

    def adjoint(self, theta=None, full_output=False):
        '''
        Obtain the gradient given input parameters using the adjoint method.
        Values of state variable are found using an univariate spline
        interpolation between two observed time points where the internal
        knots are explicitly defined.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if True, also output the full set of adjoint values (over time)

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            =================  =================================================
            key                meaning
            =================  =================================================
            'resid'            residuals given theta
            'diffLoss'         derivative of the loss function
            'gradVec'          gradient vectors
            'adjVec'           adjoint vectors
            'interpolateInfo'  info from integration over the interpolating
                               points
            'solInterpolate'   solution from the integration over the
                               interpolating points
            'tInterpolate'     interpolating time points
            =================  =================================================

        See also
        --------
        :meth:`sensitivity`

        '''

        if theta is not None:
            self._setParam(theta)

        self._ode.setParameters(self._theta)

        intName = self._ode._intName

#         # we want to construct an array with more observations
#         if self._interpolateTime is None:
#             # we have to do something about it
#             self._interpolateTime = numpy.copy(self._t)
#             for i in range(0, self._numTime-1):
#                 targetIndex = numpy.where(self._interpolateTime == self._t[i])[0][0]
#                 tTemp = numpy.linspace(self._t[i], self._t[i+1], 20)[1:-1]
#                 self._interpolateTime = numpy.insert(self._interpolateTime,
#                                                      targetIndex+1,
#                                                      tTemp)
# 
#             self._interpolateTimeIndex = list()
#             for i in range(1, self._numTime):
#                 index = numpy.where(self._interpolateTime == self._t[i])[0][0]
#                 self._interpolateTimeIndex.append(index)

        if self._interpolateTime is None:
            self._setupInterpolationTime()

        # integrate forward using the extra time points
        sAndOutInterpolate = ode_utils.integrateFuncJac(self._ode.odeT,
                                                        self._ode.JacobianT,
                                                        self._x0,
                                                        self._interpolateTime[0],
                                                        self._interpolateTime[1::],
                                                        includeOrigin=True,
                                                        full_output=full_output,
                                                        intName=intName)

        if full_output:
            solutionInterpolate = sAndOutInterpolate[0]
            outputInterpolate = sAndOutInterpolate[1]
        else:
            solutionInterpolate = sAndOutInterpolate

        # holder, assuming that the index/order is kept (and correct) in the list
        # we perform our interpolation per state and only need the functional form
        interpolateList = list()
        for j in range(0, self._numState):
            spl = scipy.interpolate.LSQUnivariateSpline(self._interpolateTime.tolist(),
                                                        solutionInterpolate[:,j],
                                                        self._t[1:-1])
            interpolateList.append(copy.deepcopy(spl))

        # find the derivative of the loss function.  they act as events
        # which are the correction to the gradient function through time
        solution = solutionInterpolate[self._interpolateTimeIndex,:]
        
        if full_output:
            g, infoDict = self._adjointGivenInterpolation(solution, interpolateList, intName, full_output)
            infoDict['interpolateInfo'] = outputInterpolate
            infoDict['solInterpolate'] = solutionInterpolate
            return g, infoDict
        else:
            return self._adjointGivenInterpolation(solution, interpolateList, intName, full_output)
#         diffLoss = self._lossObj.diffLoss(solution[:,self._stateIndex])
#         numDiffLoss = len(diffLoss)
# 
#         # finding the step size in reverse time
#         diffT = numpy.diff(self._t)
#         # print diffTRev
# 
#         # holders.  for in place insertion
#         lambdaTemp = numpy.zeros(self._numState)
#         gradList = list()
#         ga = gradList.append
#         # the last gradient value.
#         lambdaTemp[self._stateIndex] += diffLoss[-1]
#         ga(numpy.dot(self._ode.Grad(solution[-1], self._t[-1]).T, -lambdaTemp) * -diffT[-1])
# 
#         # holders if we want extra shit
#         if full_output:
#             adjVecList = list()
#             adjVecList.append(lambdaTemp)
# 
#         # integration in reverse time even though our index is going forward
#         for i in range(1, numDiffLoss):
#             # integration between two intermediate part
#             # start and the end points in time
#             tTemp = [self._t[-i-1], self._t[-i]]
# 
#             lambdaTemp[:] = ode_utils.integrateFuncJac(self._ode.adjointInterpolateT,
#                                                        self._ode.adjointInterpolateJacobianT,
#                                                        lambdaTemp, tTemp[1], tTemp[0],
#                                                        args=(interpolateList,),
#                                                        intName=intName).ravel()
# 
#             # and correction due to the "event" i.e. observed value
#             lambdaTemp[self._stateIndex] += diffLoss[-i-1]
#             # evaluate the gradient at the observed point after the correction
#             ga(numpy.dot(self._ode.Grad(solution[-i-1], tTemp[0]).T, -lambdaTemp) * -diffT[-i-1])
# 
#             if full_output:
#                 adjVecList.append(lambdaTemp)
# 
#         # the total gradient.
#         grad = numpy.array(gradList).sum(0)
# 
#         if full_output:
#             # binding the dictionaries together
#             infoDict = dict()
#             infoDict['resid'] = self._lossObj.residual(solution[:,self._stateIndex])
#             infoDict['diffLoss'] = diffLoss
#             infoDict['gradVec'] = numpy.array(gradList)
#             infoDict['adjVec'] = numpy.array(adjVecList)
#             infoDict['interpolateInfo'] = outputInterpolate
#             infoDict['solInterpolate'] = solutionInterpolate
#             infoDict['tInterpolate'] = self._interpolateTime
# 
#             return grad[self._getTargetParamIndex()], infoDict
#         else:
#             return grad[self._getTargetParamIndex()]

    #@deprecated
    def adjoint1(self, theta=None, full_output=False):
        '''
        Obtain the gradient given input parameters using the adjoint method.
        Values of state variable are found using an univariate spline
        interpolation between two observed time points without explicitly
        defining the internal knots.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if True, also output the full set of adjoint values (over time)

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information, same as
            :meth:`adjoint`

        See also
        --------
        :meth:`adjoint`

        '''

        if theta is not None:
            self._setParam(theta)

        self._ode.setParameters(self._theta)

        intName = self._ode._intName

#         interpolateTime = numpy.array([self._t[0]])
#         interpolateTimeIndex = [0]
#         numTime = len(self._t)
#         for i in range(numTime-1):
#             tTemp = numpy.linspace(self._t[i], self._t[i+1], 20)[1::]
#             interpolateTime = numpy.append(interpolateTime, tTemp)
#             interpolateTimeIndex += [len(interpolateTime)-1]
            
        if self._interpolateTime is None:
            self._setupInterpolationTime()

        # integrate forward using the extra time points
        sAndOutInterpolate = ode_utils.integrateFuncJac(self._ode.odeT,
                                                        self._ode.JacobianT,
                                                        self._x0,
                                                        self._interpolateTime[0],
                                                        self._interpolateTime[1::],
                                                        includeOrigin=True,
                                                        full_output=full_output,
                                                        intName=intName)

        if full_output:
            solutionInterpolate = sAndOutInterpolate[0]
            outputInterpolate = sAndOutInterpolate[1]
        else:
            solutionInterpolate = sAndOutInterpolate

        # holder, assuming that the index/order is kept (and correct) in the list
        # we perform our interpolation per state and only need the functional form
        interpolateList = list()
        for j in range(0, self._numState):
            spl = scipy.interpolate.InterpolatedUnivariateSpline(self._interpolateTime,
                                                                 solutionInterpolate[:,j])
            interpolateList.append(copy.deepcopy(spl))

        # find the derivative of the loss function.  they act as events
        # which are the correction to the gradient function through time
        solution = solutionInterpolate[self._interpolateTimeIndex,:]
        
        if full_output:
            g, infoDict = self._adjointGivenInterpolation(solution, interpolateList, intName, full_output)
            infoDict['interpolateInfo'] = outputInterpolate
            infoDict['solInterpolate'] = solutionInterpolate
            return g, infoDict
        else:
            return self._adjointGivenInterpolation(solution, interpolateList, intName, full_output)
        
#         diffLoss = self._lossObj.diffLoss(solution[:,self._stateIndex])
#         numDiffLoss = len(diffLoss)
# 
#         # finding the step size in reverse time
#         diffT = numpy.diff(self._t)
#         # print diffTRev
# 
#         # holders.  for in place insertion
#         lambdaTemp = numpy.zeros(self._numState)
#         gradList = list()
#         ga = gradList.append
#         # the last gradient value.
#         lambdaTemp[self._stateIndex] += diffLoss[-1]
#         ga(numpy.dot(self._ode.Grad(solution[-1], self._t[-1]).T, -lambdaTemp) * -diffT[-1])
# 
#         # holders if we want extra shit
#         if full_output:
#             adjVecList = list()
#             adjVecList.append(lambdaTemp)
# 
#         # integration in reverse time even though our index is going forward
#         for i in range(1, numDiffLoss):
#             # integration between two intermediate part
#             # start and the end points in time
#             tTemp = [self._t[-i-1], self._t[-i]]
# 
#             lambdaTemp[:] = ode_utils.integrateFuncJac(self._ode.adjointInterpolateT,
#                                                        self._ode.adjointInterpolateJacobianT,
#                                                        lambdaTemp, tTemp[1], tTemp[0],
#                                                        args=(interpolateList,),
#                                                        intName=intName).ravel()
# 
#             # and correction due to the "event" i.e. observed value
#             lambdaTemp[self._stateIndex] += diffLoss[-i-1]
#             # evaluate the gradient at the observed point after the correction
#             ga(numpy.dot(self._ode.Grad(solution[-i-1], tTemp[0]).T, -lambdaTemp) * -diffT[-i-1])
# 
#             if full_output:
#                 adjVecList.append(lambdaTemp)
# 
#         # the total gradient.
#         grad = numpy.array(gradList).sum(0)
# 
#         if full_output:
#             # binding the dictionaries together
#             infoDict = dict()
#             infoDict['resid'] = self._lossObj.residual(solution[:,self._stateIndex])
#             infoDict['diffLoss'] = diffLoss
#             infoDict['gradVec'] = numpy.array(gradList)
#             infoDict['adjVec'] = numpy.array(adjVecList)
#             infoDict['interpolateInfo'] = outputInterpolate
#             infoDict['solInterpolate'] = solutionInterpolate
#             infoDict['tInterpolate'] = self._interpolateTime
# 
#             return grad[self._getTargetParamIndex()], infoDict
#         else:
#             return grad[self._getTargetParamIndex()]

    def _setupInterpolationTime(self):
        '''
        Increase the number of output time points by putting in equally
        space points between two original time step
        '''
        interpolateTime = numpy.array([self._t[0]])
        interpolateTimeIndex = list()
        numTime = len(self._t)
        for i in range(numTime-1):
            tTemp = numpy.linspace(self._t[i], self._t[i+1], 20)[1::]
            interpolateTime = numpy.append(interpolateTime, tTemp)
            interpolateTimeIndex += [len(interpolateTime)-1]
        
        self._interpolateTime = interpolateTime
        self._interpolateTimeIndex = interpolateTimeIndex
    
    def _adjointGivenInterpolation(self, solution, interpolateList, intName, full_output=False):
        '''
        Given an interpolation of the solution of an IVP (for each state).  Compute the
        gradient via the adjoint method by a backward integration
        '''
        # find the derivative of the loss function.  they act as events
        # which are the correction to the gradient function through time
        diffLoss = self._lossObj.diffLoss(solution[:,self._stateIndex])
        numDiffLoss = len(diffLoss)

        # finding the step size in reverse time
        diffT = numpy.diff(self._t)

        # holders.  for in place insertion
        lambdaTemp = numpy.zeros(self._numState)
        gradList = list()
        ga = gradList.append
        # the last gradient value.
        lambdaTemp[self._stateIndex] += diffLoss[-1]
        ga(numpy.dot(self._ode.Grad(solution[-1], self._t[-1]).T, -lambdaTemp) * -diffT[-1])

        # holders if we want extra shit
        if full_output:
            adjVecList = list()
            adjVecList.append(lambdaTemp)

        # integration in reverse time even though our index is going forward
        for i in range(1, numDiffLoss):
            # integration between two intermediate part
            # start and the end points in time
            tTemp = [self._t[-i-1], self._t[-i]]

            lambdaTemp[:] = ode_utils.integrateFuncJac(self._ode.adjointInterpolateT,
                                                       self._ode.adjointInterpolateJacobianT,
                                                       lambdaTemp, tTemp[1], tTemp[0],
                                                       args=(interpolateList,),
                                                       intName=intName).ravel()

            # and correction due to the "event" i.e. observed value
            lambdaTemp[self._stateIndex] += diffLoss[-i-1]
            # evaluate the gradient at the observed point after the correction
            ga(numpy.dot(self._ode.Grad(solution[-i-1], tTemp[0]).T, -lambdaTemp) * -diffT[-i-1])

            if full_output:
                adjVecList.append(lambdaTemp)

        # the total gradient.
        grad = numpy.array(gradList).sum(0)
        
        if full_output:
            # binding the dictionaries together
            infoDict = dict()
            infoDict['resid'] = self._lossObj.residual(solution[:,self._stateIndex])
            infoDict['diffLoss'] = diffLoss
            infoDict['gradVec'] = numpy.array(gradList)
            infoDict['adjVec'] = numpy.array(adjVecList)
            infoDict['tInterpolate'] = self._interpolateTime

            return grad[self._getTargetParamIndex()], infoDict
        else:
            return grad[self._getTargetParamIndex()]
        
    def sensitivity(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the gradient given input parameters using forward sensitivity method.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information. Same output
            as :math:`jac`

        Notes
        -----
        It calculates the gradient by calling :meth:`jac`

        '''

        jac,output = self.jac(theta=theta, full_output=True, intName=intName)
        sens = output['sens']
        diffLoss = output['diffLoss']
        resid = output['resid']
        grad = self._sensToGradWithoutIndex(sens, diffLoss)

        if full_output:
            output['JTJ'] = self._sensToJTJWithoutIndex(sens)
            return grad, output
        else:
            return grad

    def jac(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the Jacobian of the objective function given input parameters
        using forward sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        full_output: bool, optional
            if additional output is required
        intName: str, optional
            Choice between lsoda,vode and dopri5, the three integrator provided
            by scipy.  Defaults to lsoda

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ==========  ========================================================
            key         meaning
            ==========  ========================================================
            'sens'      intermediate values over the original ode and all the
                        sensitivities, by state, parameters
            'resid'     residuals given theta
            'diffLoss'  derivative of the loss function
            ==========  ========================================================

        See also
        --------
        :meth:`sensitivity`

        '''

        if theta is not None:
            self._setParam(theta)

        self._ode.setParameters(self._theta)

        if intName is None:
            intName = self._ode._intName

        # first we want to find out the number of sensitivities required
        # add them to the initial values
        numSens =  self._numState * self._numParam
        initialStateSens = numpy.append(self._x0, numpy.zeros(numSens))

        sAndOutSens = ode_utils.integrateFuncJac(self._ode.odeAndSensitivityT,
                                                 self._ode.odeAndSensitivityJacobianT,
                                                 initialStateSens,
                                                 self._t[0], self._t[1::],
                                                 full_output=full_output,
                                                 intName=intName)

        if full_output:
            solutionSens = sAndOutSens[0]
            solutionOutput = sAndOutSens[1]
        else:
            solutionSens = sAndOutSens

        indexOut = self._getTargetParamSensIndex()

        if full_output:
            output = dict()
            output['resid'] = self._lossObj.residual(solutionSens[:,self._stateIndex])
            output['diffLoss'] = self._lossObj.diffLoss(solutionSens[:,self._stateIndex])
            output['sens'] = solutionSens
            for i in solutionOutput:
                output[i] = solutionOutput[i]

            return solutionSens[:,indexOut], output
        else:
            return solutionSens[:,indexOut]

    ############################################################
    #
    # Operators for Gradient with initial value
    #
    ############################################################

    def sensitivityIV(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the gradient given input parameters (which include
        the current guess of the initial conditions) using forward
        sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        full_output: bool, optional
            if additional output is required

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'sens'  intermediate values over the original ode and all the
                    sensitivities, by state, parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        Notes
        -----
        It calculates the gradient by calling :meth:`jacIV`

        '''

        jacIV, outputIV = self.jacIV(theta=theta, full_output=True, intName=intName)
        # the most important information! and in fact all the information we need
        # to calculate the gradient
        diffLoss = outputIV['diffLoss']
        sens = outputIV['sens']

        # grad for parameters
        grad = self._sensToGradWithoutIndex(sens, diffLoss)
        # grad for initial values
#         print sens
#         print diffLossIV
#         print diffLoss
        gradIV = self._sensToGradIVWithoutIndex(sens, diffLoss)
        # join the two
        grad = numpy.append(grad, gradIV, axis=1)

        if full_output:
            return grad, outputIV
        else:
            return grad

    def jacIV(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the Jacobian of the objective function given input parameters
        which include the current guess of the initial value using forward
        sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        full_output: bool, optional
            if additional output is required
        intName: str, optinal
            Choice between lsoda,vode and dopri5, the three integrator provided
            by scipy.  Defaults to lsoda

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'sens'  intermediate values over the original ode and all the
                    sensitivities, by state, parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivityIV`

        '''
        if theta is not None:
            self._setParamStateInput(theta)

        self._ode.setParameters(self._theta)

        if intName is None:
            intName = self._ode._intName

        # first we want to find out the number of sensitivities required
        numSens = self._numState * self._numParam
        # add them to the initial values
        initialStateSens = numpy.append(numpy.append(self._x0, numpy.zeros(numSens)),
                                        numpy.eye(self._numState).flatten())

        sAndOutSensIV = ode_utils.integrateFuncJac(self._ode.odeAndSensitivityIVT,
                                                   self._ode.odeAndSensitivityIVJacobianT,
                                                   initialStateSens,
                                                   self._t[0], self._t[1::],
                                                   full_output=full_output,
                                                   intName=intName)

        if full_output:
            solutionSensIV = sAndOutSensIV[0]
            solutionOutputIV = sAndOutSensIV[1]
        else:
            solutionSensIV = sAndOutSensIV
        # build the indexes to locate the correct parameters
        index1 = self._getTargetParamSensIndex()
        index2 = self._getTargetStateSensIndex()
        indexOut = index1+index2

        if full_output:
            output = dict()
            output['resid'] = self._lossObj.residual(solutionSensIV[:,self._stateIndex])
            output['diffLoss'] = self._lossObj.diffLoss(solutionSensIV[:,self._stateIndex])
            output['sens'] = solutionSensIV
            for i in solutionOutputIV:
                output[i] = solutionOutputIV[i]

            return solutionSensIV[:,indexOut], output
        else:
            return solutionSensIV[:,indexOut]

    ############################################################
    #
    # Operators for Hessian from ode
    #
    ############################################################

    def hessian(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the Hessian using the forward forward sensitivities.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required

        Returns
        -------
        Hessian: :class:`numpy.ndarray`
            Hessian of the objective function
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. x_{(i-1)*p + j} is the element for state i
                    and parameter j with a total of p parameters
            'hess'  intermediate values for the hessian by state, parameter,
                    parameter, i.e. x_{(i-1)*p^2 + j + k} is the element for
                    state i, parameter j and parameter k
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`

        '''
        if theta is not None:
            self._setParam(theta)

        self._ode.setParameters(self._theta)

        if intName is None:
            intName = self._ode._intName

        nS = self._numState
        nP = self._numParam
        numTime = len(self._t)

        # first we want to find out the number of initial values required to fill the
        # initial conditins
        numSens = nS * nP
        numFF = nS * nP * nP

        initialStateSens = numpy.append(self._x0, numpy.zeros(numSens + numFF))
        sAndOutAll = ode_utils.integrateFuncJac(self._ode.odeAndForwardforwardT,
                                                self._ode.odeAndForwardforwardJacobianT,
                                                initialStateSens,
                                                self._t[0], self._t[1::],
                                                full_output=full_output,
                                                intName=intName)

        if full_output:
            solutionAll = sAndOutAll[0]
            solutionOutput = sAndOutAll[1]
        else:
            solutionAll = sAndOutAll
        # the starting index for which the forward forward sensitivities are stored
        baseIndexHess = nS + nS * nP

        diffLoss = self._lossObj.diffLoss(solutionAll[:,self._stateIndex])

        H = numpy.zeros((nP, nP))

        for i in range(0, numTime-1):
            FF = ode_utils.vecToMatFF(solutionAll[i,baseIndexHess::], nS, nP)
            E = numpy.zeros(nS)
            E[self._stateIndex] += -diffLoss[i]
            H += scipy.sparse.kron(E, scipy.sparse.eye(nP)).dot(FF)

        # just the J^{T}J part of the Hessian (which is guarantee to be PSD)
        # full Hessian with the outer product gradient
        setParamIndex = self._getTargetParamIndex()
        HJTJ = H[setParamIndex][:,setParamIndex].copy()
        JTJ = self._sensToJTJWithoutIndex(solutionAll)
        HJTJ += 2*JTJ

        if full_output:
            indexOutSens = self._getTargetParamSensIndex()
            output = dict()

            output['resid'] = self._lossObj.residual(solutionAll[:,self._stateIndex])
            output['grad'] = self._sensToGradWithoutIndex(solutionAll, diffLoss)
            output['state'] = solutionAll[:,nS:(nS*(nP+1))]
            output['sens'] = solutionAll[:,indexOutSens]
            output['hess'] = solutionAll[:,baseIndexHess::]
            output['info'] = solutionOutput
            output['H'] = H
            output['JTJ'] = JTJ
            return HJTJ, output
        else:
            return HJTJ

    def jtj(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the approximation to the Hessian using the inner
        product of the Jacobian.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required

        Returns
        -------
        jtj: :class:`numpy.ndarray`
            :math:`J^{\top}J` of the objective function
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. x_{(i-1)*p + j} is the element for state i
                    and parameter j with a total of p parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`

        '''

        jac,output = self.jac(theta=theta, full_output=True, intName=intName)
        sens = output['sens']
        diffLoss = output['diffLoss']
        JTJ = self._sensToJTJWithoutIndex(sens)

        if full_output:
            sens = output['sens']
            output['grad'] = self._sensToGradWithoutIndex(sens, diffLoss)
            return JTJ, output
        else:
            return JTJ

    def fisherInformation(self, theta=None, full_output=False, intName=None):
        '''
        Obtain the Fisher information

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required

        Returns
        -------
        I: :class:`numpy.ndarray`
            :math:`I(\theta)` of the objective function
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. x_{(i-1)*p + j} is the element for state i
                    and parameter j with a total of p parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`, :meth:`jtj`

        '''

        jac,output = self.jac(theta=theta, full_output=True, intName=intName)
        sens = output['sens']
        JTJ = self._sensToJTJWithoutIndex(sens, output['resid'])

        if full_output:
            sens = output['sens']
            diffLoss = output['diffLoss']
            output['grad'] = self._sensToGradWithoutIndex(sens, diffLoss)
            return JTJ, output
        else:
            return JTJ

    ############################################################
    #
    # Other stuff related to the objective function
    #
    ############################################################

    def cost(self, theta=None):
        '''
        Find the cost/loss given time points and the corresponding
        observations.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        ode: :class:`OperateOdeModel`
            the ode class in this package
        x0: numeric
            initial time
        t0: numeric
            initial value
        t: array like
            time points where observations were made
        y: array like
            observations
        stateName: str
            the state which the observations came from

        Returns
        -------
        numeric
            sum of the residuals squared

        Notes
        -----
        Only works with a single target (state)

        See also
        --------
        :meth:`diffLoss`

        '''
        yhat = self._getSolution(theta)
        c = self._lossObj.loss(yhat)

        if c == numpy.inf:
            return numpy.nan_to_num(c)
        else:
            return c

    def diffLoss(self, theta=None):
        '''
        Find the derivative of the loss function given time points
        and the corresponding observations, with initial conditions

        Parameters
        ----------
        theta: array like
            input value of the parameters

        Returns
        -------
        :class:`numpy.ndarray`
            an array of residuals

        See also
        --------
        :meth:`cost`

        '''
        try:
            # the solution does not include the origin
            solution = self._getSolution(theta)
            return self._lossObj.diffLoss(solution)
        except Exception as e:
            print e
            print "parameters = " +str(theta)
            return numpy.nan_to_num((numpy.ones(self._y.shape)*numpy.inf))

    def residual(self, theta=None):
        '''
        Find the residuals given time points and the corresponding
        observations, with initial conditions

        Parameters
        ----------
        theta: array like
            input value of the parameters

        Returns
        -------
        :class:`numpy.ndarray`
            an array of residuals

        Notes
        -----
        Makes a direct call to :meth:`residual` using the initialized information

        See also
        --------
        :meth:`cost`, :meth:`residual`

        '''

        try:
            # the solution does not include the origin
            solution = self._getSolution(theta)
            return self._lossObj.residual(solution)
        except Exception as e:
            print e
            return numpy.nan_to_num((numpy.ones(self._y.shape)*numpy.inf))

    ############################################################
    #
    # Other crap where initial values are also parameters
    #
    ############################################################

    def costIV(self, theta=None):
        '''
        Find the cost/loss given the parameters.  :math:`\theta`
        here is assumed to include both the parameters as well as the
        initial values

        Parameters
        ----------
        theta: array like
            parameters and guess of initial values of the states

        Returns
        -------
        numeric
            sum of the residuals squared

        See also
        --------
        :meth:`residualIV`

        '''
        if theta is not None:
            self._setParamStateInput(theta)

        # try:
            # the solution does not include the origin
        solution = self._getSolution()
        return self._lossObj.loss(solution)
        # except Exception as e:
        #     print e
        #     print "parameters = " +str(theta)
        #     return numpy.nan_to_num((numpy.ones(self._y.shape)*numpy.inf))

    def diffLossIV(self, theta=None):
        '''
        Find the derivative of the loss function w.r.t. the parameters
        given time points and the corresponding observations, with
        initial conditions.

        Parameters
        ----------
        theta: array like
            parameters and initial values of the states

        Returns
        -------
        :class:`numpy.ndarray`
            an array of result

        See also
        --------
        :meth:`costIV`,:meth:`diffLoss`

        '''
        if theta is not None:
            self._setParamStateInput(theta)
        
        try:
            # the solution does not include the origin
            solution = self._getSolution()
            return self._lossObj.diffLoss(solution)
        except Exception as e:
            print e
            print "parameters = " +str(theta)
            return numpy.nan_to_num((numpy.ones(self._y.shape)*numpy.inf))

    def residualIV(self, theta=None):
        '''
        Find the residuals given time points and the corresponding
        observations, with initial conditions.

        Parameters
        ----------
        theta: array like
            parameters and initial values of the states

        Returns
        -------
        :class:`numpy.ndarray`
            an array of residuals

        Notes
        -----
        Makes a direct call to :meth:`residual` using the initialized information

        See also
        --------
        :meth:`costIV`,:meth:`residual`

        '''
        if theta is not None:
            self._setParamStateInput(theta)

        try:
            # the solution does not include the origin
            solution = self._getSolution()
            return self._lossObj.residual(solution)
        except Exception as e:
            print e
            return numpy.nan_to_num((numpy.ones(self._y.shape)*numpy.inf))

    ############################################################
    #
    # Commonly used routines in our code that are now functions
    #
    ############################################################

    def sensToGrad(self, sens, diffLoss):
        '''
        forward sensitivites to the gradient.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities
        diffLoss: array like
            derivative of the loss function

        Returns
        -------
        g: :class:`numpy.ndarray'
            gradient of the loss function
        '''
        # the number of states which we will have residuals for
        numS = len(self._stateName)
        # obviously divide through to find out the number of parameters we are inferring
        assert isinstance(sens, numpy.ndarray), "Expecting an numpy.ndarray"
        n,p = sens.shape
        assert n == len(diffLoss), ("Length of sensitivity must equal to the " +
                                    "derivative of the loss function")
                            
        numOut = p/numS # number of out parameters

        # print sens.shape
        # print diffLoss.shape
        # print self._lossObj._w.shape

        sens = numpy.reshape(sens, (n, numS, numOut), 'F')
        for j in range(numOut):
            sens[:,:,j] *= self._stateWeight

        # print diffLoss.shape
        # print sens.shape
        # print self._y.shape
        # print self._stateWeight.shape
        grad = reduce(numpy.add,map(numpy.dot, diffLoss, sens)).ravel()

        return grad

    def sensToJTJ(self, sens, resid=None):
        '''
        forward sensitivites to J^{T}J where J is the Jacobian. The
        approximation to the Hessian.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities

        Returns
        -------
        JTJ: :class:`numpy.ndarray'
            An approximation to the Hessian using the inner product
            of the Jacobian
        '''

        assert isinstance(sens, numpy.ndarray), "Expecting an numpy.ndarray"
        # the number of states which we will have residuals for
        numS = len(self._stateName)
        n,p = sens.shape
        # obviously divide through to find out the number of parameters we are inferring
        numOut = p / numS

        # define our holder accordingly
        J = numpy.zeros((numOut, numOut))
        # s = numpy.zeros((numS, numOut))

        sens = numpy.reshape(sens, (n, numS, numOut), 'F')

        for j in range(numOut):
            sens[:,:,j] *= self._stateWeight

        for i, s in enumerate(sens):
            if resid is None:
                J += numpy.dot(s.T, s)
            else:
                s1 = s * resid[i].T
                J += numpy.dot(s1.T, s1)            

        return J

    def plot(self):
        '''
        Plots the solution of all the states and the observed y values
        '''
        solution = self._getSolution(allSolution=True)
        ode_utils.plot(solution, self._observeT, self._ode._stateList,
                        self._y, self._stateName)

    def fit(self, x, lb=None, ub=None, A=None, b=None, disp=False, full_output=False):
        '''
        Find the estimates given the data given an initial guess x.  Note that there
        is no guarantee that the estimation procedure is successful.  It is
        recommended to at least supply box constraints, i.e. lower and
        upper bounds

        Parameters
        ----------
        x: array like
            an initial guess
        lb: array like
            the lower bound elementwise lb_{i} <= x_{i}
        ub: array like
            upper bound elementwise x_{i} <= ub_{i}
        A: array like
            matrix A for the inequality Ax<=b
        b: array like
            vector b for the inequality Ax<=b

        Returns
        -------
        xhat: :class:`numpy.ndarray`
            estimated value

        '''

        if lb is None or ub is None:
            if ub is None:
                ub = numpy.array([None]*len(x))
            if lb is None:
                lb = numpy.array([None]*len(x))
        else:
            if len(lb) != len(ub):
                raise InputError("Number of lower and upper bound needs to be equal")
            if len(lb) != len(x):
                raise InputError("Number of box constraints must equal to the "+
                                 " number of variables")

        boxBounds = numpy.reshape(numpy.append(lb,ub),(len(lb),2),'F')

        conList = list()

        if A is None:
            method = 'L-BFGS-B'
        else:
            if isinstance(A, numpy.ndarray):
                A = numpy.ndarray(A)
                n,p = A.shape
            if n != len(b):
                raise InputError("Number of rows in A needs to be equal to length of b "+
                                 "in the equality Ax<=b")
            if p != len(x):
                raise InputError("Number of box constraints must equal to the "+
                                 "number of variables")

            def F(a,x):
                def func(x):
                    return a.dot(x)
                return func

            for a in A: # is the row vector
                conList.append({'type':'ineq', 'fun':F(a,x)})

            method = 'SLSQP'

        if disp == True:
            callback = self.thetaCallBack
        else:
            callback = None

        res = scipy.optimize.minimize(fun = self.cost,
                                      jac = self.sensitivity,
                                      x0 = x,
                                      bounds = boxBounds,
                                      constraints = conList,
                                      method = method,
                                      callback=callback)

        if full_output:
            return res['x'], res
        else:
            return res['x']

    ############################################################
    #
    # These are "private"
    #
    ############################################################

    def _getSolution(self, theta=None, allSolution=False):
        '''
        Find the residuals given time points and the corresponding
        observations, with initial conditions
        '''

        if theta is not None:
            self._setParam(theta)

        self._ode.setParameters(self._theta)
        # TODO: is this the correct approach
        # to JacobianT what should be the return if we fail an integration

        # Note that the solution does not include the origin.  This is
        # because they do not contribute when the initial conditions are
        # given and we assume that they are accurate
        solution = ode_utils.integrateFuncJac(self._ode.odeT,
                                              self._ode.JacobianT,
                                              self._x0, self._t0,
                                              self._observeT,
                                              full_output=False,
                                              intName=self._ode._intName)
        if allSolution == True:
            return solution
        else:
            return solution[:,self._stateIndex]

    def _sensToGradWithoutIndex(self, sens, diffLoss):
        '''
        forward sensitivites to g where g is the gradient.
        Indicies obtained using information defined here
        '''
        indexOut = self._getTargetParamSensIndex()
        return self.sensToGrad(sens[:,indexOut], diffLoss)

    def _sensToGradIVWithoutIndex(self, sens, diffLoss):
        '''
        Same as sensToGradWithoutIndex above but now we also include the
        initial conditison
        '''
        indexOut = self._getTargetStateSensIndex()
        return self.sensToGrad(sens[:,indexOut], diffLoss)

    def _sensToJTJWithoutIndex(self, sens, diffLoss=None):
        '''
        forward sensitivites to J^{T}J where J is the Jacobian. The
        approximation to the Hessian.
        '''
        indexOut = self._getTargetParamSensIndex()
        return self.sensToJTJ(sens[:,indexOut], diffLoss)

    def _sensToJTJIVWithoutIndex(self, sens, diffLoss=None):
        '''
        Same as sensToJTJIVWithoutIndex above but now we also include the
        initial conditison
        '''
        indexOut = self._getTargetStateSensIndex()
        return self.sensToJTJ(sens[:,indexOut], diffLoss)


    ############################################################
    #
    # Obtain the correct index
    #
    ############################################################

    def _getTargetParamSensIndex(self):
        # as usual, locate the index of the state
        stateIndex = self._ode.getStateIndex(self._stateName)

        # build the indexes to locate the correct parameters
        indexOut = list()
        # locate the target indexes
        indexList = self._getTargetParamIndex()
        if isinstance(stateIndex, list):
            for j in stateIndex:
                for i in indexList:
                    # always ignore the first numState because they are outputs from the
                    # actual ode and not the sensitivities. Hence the +1
                    indexOut.append(j + (i+1) * self._numState)
        else:
            # else, happy times!
            for i in indexList:
                indexOut.append(stateIndex + (i+1) * self._numState)

        return numpy.sort(numpy.array(indexOut)).tolist()

    def _getTargetParamIndex(self):
        '''
        Get the indices of the targeted parameters
        '''
        # we assume that all the parameters are targets
        if self._targetParam is None:
            indexList = range(0, self._numParam)
        else:
            # only select from the list
            indexList = list()
            # note that "i" is a string here
            for i in self._targetParam:
                indexList.append(self._ode.getParamIndex(i))

        return indexList

    def _getTargetStateSensIndex(self):
        # as usual, locate the index of the state
        stateIndex = self._ode.getStateIndex(self._stateName)

        # build the indexes to locate the correct parameters
        indexOut = list()
        # locate the target indexes
        indexList = self._getTargetStateIndex()

        if isinstance(stateIndex, list):
            for j in stateIndex:
                for i in indexList:
                    # always ignore the first numState because they are outputs from the
                    # actual ode and not the sensitivities
                    indexOut.append(j + (i+1+self._numParam) * self._numState)
        else:
            # else, happy times!
            for i in indexList:
                indexOut.append(stateIndex + (i+1+self._numParam) * self._numState)

        return numpy.sort(numpy.array(indexOut)).tolist()

    def _getTargetStateIndex(self):
        '''
        Get the indices of our targeted states
        '''
        if self._targetState is None:
            indexList = range(0, self._numState)
        else:
            indexList = [self._ode.getStateIndex(i) for i in self._targetState]
            # indexList = list()
            # for i in self._targetState:
            #     indexList.append(self._ode.getStateIndex(i))

        return indexList

    def _setParamInput(self, theta):
        if self._targetParam is None:
            if len(theta) != self._numParam:
                raise InputError("Expecting input to all the parameters")
            else: # happy, standard case
                self._setParam(theta)
        else:
            if len(theta) == len(self._targetParam):
                self._unrollParam(theta)
            else:
                raise InputError("Expecting input to the parameters of length "+
                                 str(len(self._targetParam)))

    def _setParamStateInput(self, theta):
        '''
        Set both the parameters and initial conditin :math:`x_{0}`
        '''
        # print "here"
        # print len(theta)
        if self._targetParam is None and self._targetState is None:
            # we are expecting the standard case here
            if len(theta) != (self._numState+self._numParam):
                raise InputError("Expecting a guess of the initial value, use diffLoss() "+
                                 "instead for just parameter estimation")
            else:
                self._setX0(theta[-self._numState:])
                self._setParam(theta[:self._numParam])
        else:
            if self._targetParam is None:
                # this mean all the parameters or without the parameters
                if len(theta) == len(self._targetState):
                    # without parameters
                    self._unrollState(theta)
                elif len(theta) == (self._numParam+len(self._targetState)):
                    # the parameters first
                    self._setParam(theta[:self._numParam])
                    # then the states
                    # x0 = theta[-len(self._targetState):]
                    self._unrollState(theta[-len(self._targetState):])
                else:
                    raise InputError("Expecting input to all the parameters and to "+
                                    "the states with length " +str(len(self._targetState)))
            elif self._targetState is None:
                # this mean all the state or without the states
                if len(theta) == self._numParam:
                    # without the states, obviously using the wrong function call
                    raise InputError("Input has the same length as the number of parameters. "+
                                    "If the initial conditions for the states are not "+
                                    "required, use diffLoss() instead")
                elif len(theta) == (self._numState+self._numParam):
                    # all the states
                    # begin setting the information
                    self._setParam(theta[:self._numParam])
                    # then the states
                    # x0 = theta[-self._numState:]
                    self._setX0(theta[-self._numState:])
                elif len(theta) == (self._numState+len(self._targetParam)):
                    # again we have all the states
                    self._unrollParam(theta[:len(self._targetParam)])
                    # x0 = theta[-self._numState:]
                    self._setX0(theta[-self._numState:])
                else: # happy
                    raise InputError("The number of input is just plain wrong. "+
                                     "Cannot help you further")
            else:
                # we have both set of input
                if len(theta) == (len(self._targetParam)+len(self._targetState)):
                    # print "reached correct place"
                    x0 = theta[-len(self._targetState):]
                    theta = theta[:len(self._targetParam)]
                    self._unrollState(x0)
                    self._unrollParam(theta)
                else:
                    raise InputError("Input of length " +str(len(theta))+
                                    ": Expecting input to the parameters of length "+
                                    str(len(self._targetParam))+
                                    " and to the states of length "+
                                    str(len(self._targetState)))

    def _setParam(self, theta):
        '''
        Set the parameters
        '''
        # print theta
        if self._numParam == 0:
            self._theta = None
        else:
            if self._targetParam is not None:
                thetaDict = dict()
                if len(self._targetParam) > 1:
                    if len(theta) != len(self._targetParam):
                        raise InputError("Input length = " +str(len(theta))+
                                         " but we expect " +str(len(self._targetParam)))
                    # begin to construct our dictionary
                    for i in range(0, len(theta)):
                        thetaDict[self._targetParam[i]] = theta[i]
                else:
                    if ode_utils.isNumeric(theta):
                        thetaDict[self._targetParam[0]] = theta
                    elif len(theta) > 1:
                        raise InputError("Input length = " +str(len(theta))+
                                         " but we only have one parameter")
                    else:
                        thetaDict[self._targetParam[0]] = theta[0]
                self._theta = thetaDict
            else:
                # conver to something sensible
                theta = ode_utils.checkArrayType(theta)
                self._theta = numpy.copy(theta)

    def _setWeight(self, n, p, w):
        # note that we NEVER scale the weights
        # also note that we can use the weights as a control
        # with normalized input

        w = ode_utils.checkArrayType(w)
        if len(w) == w.size:
            m = len(w)
            q = 1
        else:
            m, q = w.shape

        # print n,p
        # print m,q

        if p == q:
            if n == m:
                self._stateWeight = w
            elif m == 1:
                self._stateWeight = numpy.ones((n,p)) * w
            else:
                raise InputError("Number of input weights is not equal "+
                                "to the number of observations")
        elif p == m:
            if q == 1:
                self._stateWeight = numpy.ones((n,p)) * w
            else:
                raise InputError("Number of input weights is not equal "+
                                 "to number of states")
        else:
            if q == 1 and m == 1:
                self._stateWeight = numpy.ones((n,p)) * w                
            else:
                raise InputError("Number of input weights differs from the "+
                                 "number of observations")

    def _setX0(self, x0):
        '''
        Set the initial value, pretty much only used when we are
        dealing with estimating the initial value as well
        '''
        x0 = ode_utils.checkArrayType(x0)
        self._x0 = numpy.copy(x0)

    def _setLossType(self):
        '''
        we set the loss type desired. This is the method that will
        be override in the module odeLoss.  Basically, all other
        operations remains but this will change.
        '''
        self._lossObj = Square(self._y, self._stateWeight)
        return self._lossObj

    def _unrollParam(self, theta):
        '''
        The difference between this and _setParam is that this method
        only works if the self._theta exist, i.e. _setParam has been
        invoked previously
        '''
        if self._targetParam is not None:
            # both are dictionary, straight copy over
            if isinstance(theta, dict):
                for k, v in theta.iteritems():
                    self._theta[k] = v
            else:
                # print "fake"
                # theta only contains the value
                for i in range(0, len(theta)):
                    # unroll the name of the parameters
                    paramStr = self._targetParam[i]
                    self._theta[paramStr] = theta[i]
        else: # it is none, we swap all the values
            if isinstance(self._theta, dict):
                i = 0
                for k, v in self._theta.iteritems():
                    self._theta[k] = theta[i]
                    i += 1
            else:
                for i in range(0, len(theta)):
                    self._theta[i] = theta[i]

    def _unrollState(self, x0):
        '''
        If the target state are not entered in sequence, then we need
        to adjust and assign the correct index
        '''
        for i in range(0, len(self._targetState)):
            index = self._ode.getStateIndex(self._targetState[i])
            self._x0[index] = x0[i]

    def thetaCallBack(self, x):
        '''
        Print x, the parameters
        '''
        print x

    def thetaCallBack2(self, x, f):
        '''
        Print x and f where x is the parameter of interest
        and f is the objective function

        Parameters
        ----------
        x:
            parameters
        f:
            f(x)
        '''
        print "f(x) = " +str(f)+ " ; x = " + str(x)

    def _selfInner(self, A):
        return A.T.dot(A)
