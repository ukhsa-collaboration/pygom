"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    To place everything about estimating the parameters of an ode model
    under square loss in one single module.  Focus on the standard local
    method which means obtaining the gradient and Hessian.

"""

#__all__ = [] # don't really want to export this

import copy
import functools
from numbers import Number

import numpy as np
import scipy.sparse
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import minimize

from pygom.loss.loss_type import Square
from pygom.model import ode_utils
from pygom.model._model_errors import InputError
from pygom.model.ode_variable import ODEVariable

class BaseLoss(object):
    """
    This contains the base that stores all the information of an ode.

    Parameters
    ----------
    theta: array like
        input value of the parameters
    ode: :class:`DeterministicOde`
        the ode class in this package
    x0: numeric
        initial time
    t0: numeric
        initial value
    t: array like
        time points where observations were made
    y: array like
        observations
    state_name: str
        the state which the observations came from
    state_weight: array like
        weight for the observations
    target_param: str or array like
        parameters that are not fixed
    target_state: str or array like
        states that are not fixed, applicable only when the initial
        values are also of interest
    """
    def __init__(self, theta, ode,
                 x0, t0,
                 t, y,
                 state_name, state_weight=None,
                 target_param=None, target_state=None):

        ### Execute all the checks first

        # conversion into np
        t = ode_utils.check_array_type(t)
        y = ode_utils.check_array_type(y)

        if state_weight is None:
            state_weight = 1.0

        if len(y) == y.size:
            y = y.flatten()
            n, p = len(y), 1
        else:
            n, p = y.shape

        assert len(t) == n, "Number of observations and time must be equal"

        ## TODO: think about whether this should be a copy
        ## there are pros and cons with referencing or copy
        ## if we copy, we isolate the ode so we can make a
        ## distributed/parallel estimate but it is easier to diagnose problems
        ## when we don't copy and also make use of the plotting methods
        ## because the parameters are now basically just a pointer that is
        ## continuously updated
        self._ode = ode

        # We are making a shitty check here because I screwed up (sort of)
        # Should have been a base class where we do not have the target_param
        # and target_state and another class extending it.  The only problem of
        # that is the lost of ability to make faster calculation, which is not
        # even possible now because of how OperateOdeModel works.  Ideally,
        # OperateOdeModel will take in the target_param in a way so that the
        # gradient information is only computed on those targeted instead of
        # computing the full vector before extracting the relevant elements.
        # Basically, it will require a lot of work to make things sync and
        # that is too much effort and time which I do not have
        if self._ode.parameters is None:
            if self._ode.num_param != 0:
                # note that this is necessary because we want to make sure that
                # it is possible to only estimate part of the full parameter set
                raise RuntimeError("Set the parameters of the ode first")
        else:
            try:
                self._ode.initial_values = (x0, t0)
                solution = self._ode.integrate2(t)
            except Exception as e:
                # print(e)
                if t0 == t[1]:
                    raise InputError("First time point t[1] is equal to t0")
                else:
                    raise InputError("ode not initialized properly or " +
                                     "unable to integrate using the initial " +
                                     "values provided")

        # Information
        self._num_param = self._ode.num_param
        self._num_state = self._ode.num_state

        ### We wish to know whether we are dealing with a multiobjective problem

        # decide whether we are working on a restricted set
        # the target parameters
        if target_param is None:
            self._targetParam = None
        else:
            self._targetParam = ode_utils.str_or_list(target_param)

        if target_state is None:
            self._targetState = None
        else:
            self._targetState = ode_utils.str_or_list(target_state)

        # check stuff
        # if you are trying to go through this, I apologize
        if state_name is None:
            # then if
            if solution.shape[1] == p:
                state_name = [str(i) for i in self._ode._iterStateList()]
                self._setWeight(n, p, state_weight)
            else:
                raise InputError("Expecting the name of states " +
                                 "for the observations")
        elif isinstance(state_name, (str, list, tuple)):
            if isinstance(state_name, str):
                state_name = [state_name]

            assert p == len(state_name), "len(state_name) and len(y[0]) not equal"
            self._setWeight(n, p, state_weight)
        else:
            raise InputError("State name should be str or of type list/tuple")

        # if self._stateWeight is not None:
        if np.any(self._stateWeight <= 0):
            raise InputError("Weights should be strictly positive")

        # finish ordering information
        # now get the index of target states
        self._stateName = state_name
        self._stateIndex = self._ode.get_state_index(self._stateName)
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
        self._t = np.insert(t, 0, t0)
        # and length
        self._numTime = len(self._t)

        # interpolating information
        self._interpolateTime = None
        self._interpolateTimeIndex = None

        # TODO: optimal weight in terms of Pareto front from a
        # multiobjective optimization perspective
        self._lossObj = self._setLossType()

        # final check
        if self._t is None or self._y is None or self._stateName is None:
            raise InputError("Error without data currently not implemented")

    def _get_model_str(self):

        if isinstance(self._theta, dict):
            _theta = list(self._theta.values())
        else:
            _theta = self._theta.tolist()
        model_str = "(%s, %s, %s, %s, %s, %s, %s" % (_theta,
                                                    self._ode,
                                                    self._x0.tolist(),
                                                    self._t0,
                                                    self._observeT.tolist(),
                                                    self._y.tolist(),
                                                    self._stateName)
        if self._stateWeight is not None:
            model_str += ", %s" % self._stateWeight.tolist()
        if self._targetParam is not None:
            model_str += ", %s" % self._targetParam
        if self._targetState is not None:
            model_str += ", %s" % self._targetState
        return model_str + ")"

    ############################################################
    #
    # Gradient operators
    #
    ############################################################

    def gradient(self, theta=None, full_output=False):
        """
        Returns the gradient calculated by solving the forward sensitivity
        equation.  Identical to :meth:`sensitivity` without the choice of
        integration method

        See Also
        --------
        :meth:`sensitivity`
        """
        return self.sensitivity(theta, full_output)

    def adjoint(self, theta=None, full_output=False):
        """
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
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            =================  =================================================
            key                meaning
            =================  =================================================
            'resid'            residuals given theta
            'diff_loss'         derivative of the loss function
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

        """

        if theta is not None:
            self._setParam(theta)

        self._ode.parameters = self._theta

        if self._interpolateTime is None:
            self._setupInterpolationTime()

        # integrate forward using the extra time points
        f = ode_utils.integrateFuncJac
        s_and_i = f(self._ode.ode_T,
                               self._ode.jacobian_T,
                               self._x0,
                               self._interpolateTime[0],
                               self._interpolateTime[1::],
                               includeOrigin=True,
                               full_output=full_output,
                               method=self._ode._intName)

        if full_output:
            sol = s_and_i[0]
            out = s_and_i[1]
        else:
            sol = s_and_i

        # holder, assuming that the index/order is kept (and correct) in
        # the list we perform our interpolation per state and only need
        # the functional form
        interpolate_list = list()
        for j in range(self._num_state):
            spl = LSQUnivariateSpline(self._interpolateTime.tolist(),
                                      sol[:, j],
                                      self._t[1:-1])
            interpolate_list.append(copy.deepcopy(spl))

        # find the derivative of the loss function.  they act as events
        # which are the correction to the gradient function through time
        solution = sol[self._interpolateTimeIndex, :]

        if full_output:
            g, info_dict = self._adjointGivenInterpolation(solution,
                                                          interpolate_list,
                                                          self._ode._intName,
                                                          full_output)
            info_dict['interpolateInfo'] = out
            info_dict['solInterpolate'] = sol
            return g, info_dict
        else:
            return self._adjointGivenInterpolation(solution, interpolate_list,
                                                   self._ode._intName,
                                                   full_output)

    def _setupInterpolationTime(self):
        """
        Increase the number of output time points by putting in equally
        space points between two original time step
        """
        interpolate_time = np.array([self._t[0]])
        interpolate_index = list()
        num_time = len(self._t)
        for i in range(num_time - 1):
            tTemp = np.linspace(self._t[i], self._t[i+1], 20)[1::]
            interpolate_time = np.append(interpolate_time, tTemp)
            interpolate_index += [len(interpolate_time) - 1]

        self._interpolateTime = interpolate_time
        self._interpolateTimeIndex = interpolate_index

    def _adjointGivenInterpolation(self, solution, interpolateList,
                                   method, full_output=False):
        """
        Given an interpolation of the solution of an IVP (for each state).
        Compute the gradient via the adjoint method by a backward integration
        """
        # find the derivative of the loss function.  they act as events
        # which are the correction to the gradient function through time
        diff_loss = self._lossObj.diff_loss(solution[:,self._stateIndex])
        num_diff_loss = len(diff_loss)

        # finding the step size in reverse time
        diff_t = np.diff(self._t)

        # holders.  for in place insertion
        lambda_temp = np.zeros(self._num_state)
        grad_list = list()
        ga = grad_list.append
        # the last gradient value.
        lambda_temp[self._stateIndex] += diff_loss[-1]
        ga(np.dot(self._ode.grad(solution[-1], self._t[-1]).T,
                  -lambda_temp)*-diff_t[-1])

        # holders if we want extra shit
        if full_output:
            adj_vec_list = list()
            adj_vec_list.append(lambda_temp)

        # integration in reverse time even though our index is going forward
        f = ode_utils.integrateFuncJac
        for i in range(1, num_diff_loss):
            # integration between two intermediate part
            # start and the end points in time
            tTemp = [self._t[-i-1], self._t[-i]]

            lambda_temp[:] = f(self._ode.adjoint_interpolate_T,
                              self._ode.adjoint_interpolate_jacobian_T,
                              lambda_temp, tTemp[1], tTemp[0],
                              args=(interpolateList,),
                              method=method).ravel()

            # and correction due to the "event" i.e. observed value
            lambda_temp[self._stateIndex] += diff_loss[-i-1]
            # evaluate the gradient at the observed point after the correction
            ga(np.dot(self._ode.grad(solution[-i-1], tTemp[0]).T,
                         -lambda_temp)*-diff_t[-i-1])

            if full_output:
                adj_vec_list.append(lambda_temp)

        # the total gradient.
        grad = np.array(grad_list).sum(0)

        if full_output:
            # binding the dictionaries together
            infoDict = dict()
            infoDict['resid'] = self._lossObj.residual(solution[:,self._stateIndex])
            infoDict['diff_loss'] = diff_loss
            infoDict['gradVec'] = np.array(grad_list)
            infoDict['adjVec'] = np.array(adj_vec_list)
            infoDict['tInterpolate'] = self._interpolateTime

            return grad[self._getTargetParamIndex()], infoDict
        else:
            return grad[self._getTargetParamIndex()]

    def sensitivity(self, theta=None, full_output=False, method=None):
        """
        Obtain the gradient given input parameters using forward
        sensitivity method.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required
        method: str, optional
            what method to use in the integrator

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information. Same output
            as :meth:`jac`

        Notes
        -----
        It calculates the gradient by calling :meth:`jac`

        """
        if full_output:
            _jac, output = self.jac(theta=theta, full_output=True, method=method)
            sens = output['sens']
            diff_loss = output['diff_loss']
            grad = self._sensToGradWithoutIndex(sens, diff_loss)
            output['JTJ'] = self._sensToJTJWithoutIndex(sens)

            return grad, output
        else:
            _jac, sens = self.jac(theta=theta, sens_output=True, full_output=False, method=method)
            i = self._stateIndex
            diff_loss = self._lossObj.diff_loss(sens[:,i])
            grad = self._sensToGradWithoutIndex(sens, diff_loss)

            return grad

    def jac(self, theta=None, sens_output=False, full_output=False, method=None):
        """
        Obtain the Jacobian of the objective function given input parameters
        using forward sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        sens_output: bool, optional
            whether the full sensitivities is required; full_output overrides this
            option when true
        full_output: bool, optional
            if additional output is required
        method: str, optional
            Choice between lsoda, vode and dopri5, the three integrator
            provided by scipy.  Defaults to lsoda.

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ===========  =======================================================
            key          meaning
            ===========  =======================================================
            'sens'       intermediate values over the original ode and all the
                         sensitivities, by state, parameters
            'resid'      residuals given theta
            'diff_loss'  derivative of the loss function
            ===========  =======================================================

        See also
        --------
        :meth:`sensitivity`

        """

        if theta is not None:
            self._setParam(theta)

        self._ode.parameters = self._theta

        if method is None:
            method = self._ode._intName

        # first we want to find out the number of sensitivities required
        # add them to the initial values
        num_sens =  self._num_state*self._num_param
        init_state_sens = np.append(self._x0, np.zeros(num_sens))

        f = ode_utils.integrateFuncJac

        index_out = self._getTargetParamSensIndex()

        if full_output:
            s_sens = f(self._ode.ode_and_sensitivity_T,
                       self._ode.ode_and_sensitivity_jacobian_T,
                       init_state_sens,
                       self._t[0], self._t[1::],
                       full_output=full_output,
                       method=method)
            sol_sens = s_sens[0]
            sol_out = s_sens[1]

            output = dict()
            i = self._stateIndex
            output['resid'] = self._lossObj.residual(sol_sens[:, i])
            output['diff_loss'] = self._lossObj.diff_loss(sol_sens[:, i])
            output['sens'] = sol_sens
            for i in sol_out:
                output[i] = sol_out[i]

            return sol_sens[:,index_out], output
        else:
            sol_sens = f(self._ode.ode_and_sensitivity_T,
                         self._ode.ode_and_sensitivity_jacobian_T,
                         init_state_sens,
                         self._t[0], self._t[1::],
                         method=method)

            if sens_output:
                return sol_sens[:, index_out], sol_sens
            else:
                return sol_sens[:,index_out]

    ############################################################
    #
    # Operators for Gradient with initial value
    #
    ############################################################

    def sensitivityIV(self, theta=None, full_output=False, method=None):
        """
        Obtain the gradient given input parameters (which include the current
        guess of the initial conditions) using forward sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        full_output: bool, optional
            if additional output is required
        method: str, optional
            what method to use in the integrator

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient
        infodict : dict, only returned if full_output=True
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
        """

        if full_output:
            _jac_iv, output_iv = self.jacIV(theta=theta,
                                            full_output=True,
                                            method=method)
            # the most important information! and in fact all the information
            # we need to calculate the gradient
            diff_loss = output_iv['diff_loss']
            sens = output_iv['sens']

            grad = self._sensToGradWithoutIndex(sens, diff_loss)
            grad_iv = self._sensToGradIVWithoutIndex(sens, diff_loss)
            grad = np.append(grad, grad_iv)

            return grad, output_iv
        else:
            _sol_iv, sens = self.jacIV(theta=theta,
                                       sens_output=True,
                                       full_output=False,
                                       method=method)

            i = self._stateIndex
            diff_loss = self._lossObj.diff_loss(sens[:, i])

            # grad for parameters and the initial values. Then join the two
            grad = self._sensToGradWithoutIndex(sens, diff_loss)
            grad_iv = self._sensToGradIVWithoutIndex(sens, diff_loss)
            grad = np.append(grad, grad_iv)

            return grad

    def jacIV(self, theta=None, sens_output=False, full_output=False, method=None):
        """
        Obtain the Jacobian of the objective function given input parameters
        which include the current guess of the initial value using forward
        sensitivity method.

        Parameters
        ----------
        theta: array like, optional
            input value of the parameters
        sens_output: bool, optional
            whether the full sensitivities is required; full_output overrides this
            option when true
        full_output: bool, optional
            if additional output is required
        method: str, optional
            Choice between lsoda, vode and dopri5, the three integrator
            provided by scipy.  Defaults to lsoda

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output=True
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
        """
        if theta is not None:
            self._setParamStateInput(theta)

        self._ode.parameters = self._theta

        if method is None:
            method = self._ode._intName

        # first we want to find out the number of sensitivities required
        num_sens = self._num_state*self._num_param
        # add them to the initial values
        initial_state_sens = np.append(np.append(self._x0, np.zeros(num_sens)),
                                        np.eye(self._num_state).flatten())

        f = ode_utils.integrateFuncJac

        # build the indexes to locate the correct parameters
        index1 = self._getTargetParamSensIndex()
        index2 = self._getTargetStateSensIndex()
        index_out = index1 + index2

        if full_output:
            s_iv = f(self._ode.ode_and_sensitivityIV_T,
                     self._ode.ode_and_sensitivityIV_jacobian_T,
                     initial_state_sens,
                     self._t[0], self._t[1::],
                     full_output=full_output,
                     method=method)
            sol_iv = s_iv[0]
            output_iv = s_iv[1]

            output = dict()
            i = self._stateIndex
            output['resid'] = self._lossObj.residual(sol_iv[:, i])
            output['diff_loss'] = self._lossObj.diff_loss(sol_iv[:, i])
            output['sens'] = sol_iv
            for i in output_iv:
                output[i] = output_iv[i]

            return sol_iv[:, index_out], output
        else:
            sol_iv = f(self._ode.ode_and_sensitivityIV_T,
                       self._ode.ode_and_sensitivityIV_jacobian_T,
                       initial_state_sens,
                       self._t[0], self._t[1::],
                       method=method)

            if sens_output:
                return sol_iv[:, index_out], sol_iv
            else:
                return sol_iv[:, index_out]

    ############################################################
    #
    # Operators for Hessian from ode
    #
    ############################################################

    def hessian(self, theta=None, full_output=False, method=None):
        """
        Obtain the Hessian using the forward forward sensitivities.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required
        method: str, optional
            what method to use in the integrator

        Returns
        -------
        Hessian: :class:`numpy.ndarray`
            Hessian of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. :math:`x_{(i-1)p + j}` is the element for
                    state :math:`i` and parameter :math:`j` with a total of
                    :math:`p` parameters
            'hess'  intermediate values for the hessian by state, parameter,
                    parameter, i.e. :math:`x_{(i-1)p^{2} + j + k}` is the
                    element for state :math:`i`, parameter :math:`j` and
                    parameter :math:`k`
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`

        """
        if theta is not None:
            self._setParam(theta)

        self._ode.parameters = self._theta

        if method is None:
            method = self._ode._intName

        nS = self._num_state
        nP = self._num_param
        num_time = len(self._t)

        # first we want to find out the number of initial values required to
        # fill the initial conditins
        num_sens = nS*nP
        num_ff = nS*nP*nP

        initial_state_sens = np.append(self._x0, np.zeros(num_sens + num_ff))

        f = ode_utils.integrateFuncJac
        s_out_all = f(self._ode.ode_and_forwardforward_T,
                      self._ode.ode_and_forwardforward_jacobian_T,
                      initial_state_sens,
                      self._t[0], self._t[1::],
                      full_output=full_output,
                      method=method)

        if full_output:
            solution_all = s_out_all[0]
            solution_output = s_out_all[1]
        else:
            solution_all = s_out_all
        # the starting index for which the forward forward sensitivities
        # are stored
        base_index_hess = nS + nS*nP

        diff_loss = self._lossObj.diff_loss(solution_all[:,self._stateIndex])

        H = np.zeros((nP, nP))

        for i in range(num_time - 1):
            FF = ode_utils.vecToMatFF(solution_all[i,base_index_hess::], nS, nP)
            E = np.zeros(nS)
            E[self._stateIndex] += -diff_loss[i]
            H += scipy.sparse.kron(E, scipy.sparse.eye(nP)).dot(FF)

        # just the J^{\top}J part of the Hessian (which is guarantee to be PSD)
        # full Hessian with the outer product gradient
        param_idx = self._getTargetParamIndex()
        HJTJ = H[param_idx][:, param_idx].copy()
        JTJ = self._sensToJTJWithoutIndex(solution_all)
        HJTJ += 2*JTJ

        if full_output:
            sens_idx = self._getTargetParamSensIndex()
            output = dict()

            i = self._stateIndex
            output['resid'] = self._lossObj.residual(solution_all[:, i])
            output['grad'] = self._sensToGradWithoutIndex(solution_all, diff_loss)
            output['state'] = solution_all[:, nS:(nS*(nP+1))]
            output['sens'] = solution_all[:, sens_idx]
            output['hess'] = solution_all[:, base_index_hess::]
            output['info'] = solution_output
            output['H'] = H
            output['JTJ'] = JTJ
            return HJTJ, output
        else:
            return HJTJ

    def jtj(self, theta=None, full_output=False, method=None):
        """
        Obtain the approximation to the Hessian using the inner
        product of the Jacobian.

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required
        method: str, optional
            what method to use in the integrator

        Returns
        -------
        jtj: :class:`numpy.ndarray`
            :math:`J^{\\top}J` of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. :math:`x_{(i-1)p + j}` is the element for
                    state :math:`i` and parameter :math:`j` with a total of
                    :math:`p` parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`
        """

        _jac, output = self.jac(theta=theta, full_output=True, method=method)
        sens = output['sens']
        JTJ = self._sensToJTJWithoutIndex(sens)

        if full_output:
            diff_loss = output['diff_loss']
            output['grad'] = self._sensToGradWithoutIndex(sens, diff_loss)
            return JTJ, output
        else:
            return JTJ

    def fisher_information(self, theta=None, full_output=False, method=None):
        """
        Obtain the Fisher information

        Parameters
        ----------
        theta: array like
            input value of the parameters
        full_output: bool
            if additional output is required
        method: str, optional
            what method to use in the integrator

        Returns
        -------
        I: :class:`numpy.ndarray`
            :math:`I(\\theta)` of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ======= ============================================================
            key     meaning
            ======= ============================================================
            'state' intermediate values for the state (original ode)
            'sens'  intermediate values for the sensitivities by state,
                    parameters, i.e. :math:`x_{(i-1)p + j}` is the element for
                    state :math:`i` and parameter :math:`j` with a total of
                    :math:`p` parameters
            'resid' residuals given theta
            'info'  output from the integration
            ======= ============================================================

        See also
        --------
        :meth:`sensitivity`, :meth:`jtj`

        """

        _jac, output = self.jac(theta=theta, full_output=True, method=method)
        sens = output['sens']
        JTJ = self._sensToJTJWithoutIndex(sens, output['resid'])

        if full_output:
            sens = output['sens']
            diffLoss = output['diff_loss']
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
        """
        Find the cost/loss given time points and the corresponding
        observations.

        Parameters
        ----------
        theta: array like
            input value of the parameters

        Returns
        -------
        numeric
            sum of the residuals squared

        Notes
        -----
        Only works with a single target (state)

        See also
        --------
        :meth:`diff_loss`

        """
        yhat = self._getSolution(theta)
        c = self._lossObj.loss(yhat)

        return np.nan_to_num(c) if c == np.inf else c

    def diff_loss(self, theta=None):
        """
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
        """
        try:
            # the solution does not include the origin
            solution = self._getSolution(theta)
            return self._lossObj.diff_loss(solution)
        except Exception as e:
            # print(e)
            # print("parameters = " +str(theta))
            return np.nan_to_num((np.ones(self._y.shape)*np.inf))

    def residual(self, theta=None):
        """
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
        Makes a direct call to initialized loss object which has a
        method called residual

        See also
        --------
        :meth:`cost`
        """

        try:
            # the solution does not include the origin
            solution = self._getSolution(theta)
            return self._lossObj.residual(solution)
        except Exception as e:
            # print(e)
            return np.nan_to_num((np.ones(self._y.shape)*np.inf))

    ############################################################
    #
    # Other crap where initial values are also parameters
    #
    ############################################################

    def costIV(self, theta=None):
        """
        Find the cost/loss given the parameters. The input theta
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

        """
        if theta is not None:
            self._setParamStateInput(theta)

        solution = self._getSolution()
        return self._lossObj.loss(solution)

    def diff_lossIV(self, theta=None):
        """
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
        :meth:`costIV`, :meth:`diff_loss`

        """
        if theta is not None:
            self._setParamStateInput(theta)

        try:
            # the solution does not include the origin
            solution = self._getSolution()
            return self._lossObj.diff_loss(solution)
        except Exception as e:
            # print(e)
            # print("parameters = " + str(theta))
            return np.nan_to_num((np.ones(self._y.shape)*np.inf))

    def residualIV(self, theta=None):
        """
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
        Makes a direct call to :meth:`residual` using the
        initialized information

        See also
        --------
        :meth:`costIV`, :meth:`residual`

        """
        if theta is not None:
            self._setParamStateInput(theta)

        try:
            # the solution does not include the origin
            solution = self._getSolution()
            return self._lossObj.residual(solution)
        except Exception as e:
            # print(e)
            return np.nan_to_num((np.ones(self._y.shape)*np.inf))

    ############################################################
    #
    # Commonly used routines in our code that are now functions
    #
    ############################################################

    def sens_to_grad(self, sens, diff_loss):
        """
        Forward sensitivites to the gradient.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities
        diff_loss: array like
            derivative of the loss function

        Returns
        -------
        g: :class:`numpy.ndarray`
            gradient of the loss function
        """
        # the number of states which we will have residuals for
        num_s = len(self._stateName)

        assert isinstance(sens, np.ndarray), "Expecting an np.ndarray"
        n, p = sens.shape
        assert n == len(diff_loss), ("Length of sensitivity must equal to " +
                                     "the derivative of the loss function")

        # Divide through to obtain the number of parameters we are inferring
        num_out = int(p/num_s) # number of out parameters

        sens = np.reshape(sens, (n, num_s, num_out), 'F')
        for j in range(num_out):
            sens[:, :, j] *= self._stateWeight

        grad = functools.reduce(np.add,map(np.dot, diff_loss, sens)).ravel()

        return grad

    def sens_to_jtj(self, sens, resid=None):
        """
        forward sensitivites to :math:`J^{\\top}J` where :math:`J` is the
        Jacobian. The approximation to the Hessian.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities
        resid: :class:`numpy.ndarray`, optional
            the residuals corresponding to the input sens

        Returns
        -------
        JTJ: :class:`numpy.ndarray`
            An approximation to the Hessian using the inner product
            of the Jacobian
        """

        assert isinstance(sens, np.ndarray), "Expecting an np.ndarray"
        # the number of states which we will have residuals for
        num_s = len(self._stateName)
        n, p = sens.shape
        # obviously divide through to find out the number of parameters
        # we are inferring
        num_out = int(p/num_s)

        # define our holder accordingly
        J = np.zeros((num_out, num_out))
        # s = np.zeros((numS, numOut))

        sens = np.reshape(sens, (n, num_s, num_out), 'F')

        for j in range(num_out):
            sens[:,:,j] *= self._stateWeight

        for i, s in enumerate(sens):
            if resid is None:
                J += np.dot(s.T, s)
            else:
                s1 = s*resid[i].T
                J += np.dot(s1.T, s1)

        return J

    def plot(self):
        """
        Plots the solution of all the states and the observed y values
        """
        solution = self._getSolution(all_solution=True)
        ode_utils.plot_det(solution, self._observeT, self._ode._stateList,
                           self._y, self._stateName)

    def fit(self, x, lb=None, ub=None, A=None, b=None,
            disp=False, full_output=False):
        """
        Find the estimates given the data and an initial guess :math:`x`.
        Note that there is no guarantee that the estimation procedure is
        successful.  It is recommended to at least supply box constraints,
        i.e. lower and upper bounds

        Parameters
        ----------
        x: array like
            an initial guess
        lb: array like
            the lower bound elementwise :math:`lb_{i} <= x_{i}`
        ub: array like
            upper bound elementwise :math:`x_{i} <= ub_{i}`
        A: array like
            matrix :math:`A` for the inequality :math:`Ax<=b`
        b: array like
            vector :math:`b` for the inequality :math:`Ax<=b`

        Returns
        -------
        xhat: :class:`numpy.ndarray`
            estimated value

        """

        if lb is None or ub is None:
            if ub is None:
                ub = np.array([None]*len(x))
            if lb is None:
                lb = np.array([None]*len(x))
        else:
            if len(lb) != len(ub):
                raise InputError("Number of lower and upper bounds " +
                                 "needs to be equal")
            if len(lb) != len(x):
                raise InputError("Number of box constraints must equal to " +
                                 "the number of variables")

        box_bounds = np.reshape(np.append(lb, ub), (len(lb), 2), 'F')

        con_list = list()

        if A is None:
            method = 'L-BFGS-B'
        else:
            if isinstance(A, np.ndarray):
                A = np.ndarray(A)
                n,p = A.shape
            if n != len(b):
                raise InputError("Number of rows in A needs to be equal to " +
                                 "length of b in the equality Ax<=b")
            if p != len(x):
                raise InputError("Number of box constraints must equal to " +
                                 "the number of variables")

            def F(a, x):
                def func(x):
                    return a.dot(x)
                return func

            for a in A: # is the row vector
                con_list.append({'type': 'ineq', 'fun': F(a,x)})

            method = 'SLSQP'

        if disp == True:
            callback = self.thetaCallBack
        else:
            callback = None

        res = minimize(fun=self.cost,
                       jac=self.sensitivity,
                       x0=x,
                       bounds=box_bounds,
                       constraints=con_list,
                       method=method,
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

    def _getSolution(self, theta=None, all_solution=False):
        """
        Find the residuals given time points and the corresponding
        observations, with initial conditions
        """

        if theta is not None:
            self._setParam(theta)

        self._ode.parameters = self._theta
        # TODO: is this the correct approach
        # to jacobian_T what should be the return if we fail an integration

        # Note that the solution does not include the origin.  This is
        # because they do not contribute when the initial conditions are
        # given and we assume that they are accurate
        solution = ode_utils.integrateFuncJac(self._ode.ode_T,
                                              self._ode.jacobian_T,
                                              self._x0, self._t0,
                                              self._observeT,
                                              full_output=False,
                                              method=self._ode._intName)
        if all_solution:
            return solution
        else:
            return solution[:, self._stateIndex]

    def _sensToGradWithoutIndex(self, sens, diffLoss):
        """
        forward sensitivites to g where g is the gradient.
        Indicies obtained using information defined here
        """
        index_out = self._getTargetParamSensIndex()
        return self.sens_to_grad(sens[:, index_out], diffLoss)

    def _sensToGradIVWithoutIndex(self, sens, diffLoss):
        """
        Same as sensToGradWithoutIndex above but now we also include the
        initial conditions.
        """
        index_out = self._getTargetStateSensIndex()
        return self.sens_to_grad(sens[:, index_out], diffLoss)

    def _sensToJTJWithoutIndex(self, sens, diffLoss=None):
        """
        forward sensitivites to :math:`J^{\\top}J: where :math:`J` is
        the Jacobian. The approximation to the Hessian.
        """
        index_out = self._getTargetParamSensIndex()
        return self.sens_to_jtj(sens[:, index_out], diffLoss)

    def _sensToJTJIVWithoutIndex(self, sens, diffLoss=None):
        """
        Same as sensToJTJIVWithoutIndex above but now we also include the
        initial conditions.
        """
        index_out = self._getTargetStateSensIndex()
        return self.sens_to_jtj(sens[:, index_out], diffLoss)


    ############################################################
    #
    # Obtain the correct index
    #
    ############################################################

    def _getTargetParamSensIndex(self):
        # as usual, locate the index of the state
        state_index = self._ode.get_state_index(self._stateName)

        # build the indexes to locate the correct parameters
        index_out = list()
        # locate the target indexes
        index_list = self._getTargetParamIndex()
        if isinstance(state_index, list):
            for j in state_index:
                for i in index_list:
                    # always ignore the first numState because they are
                    # outputs from the actual ode and not the sensitivities.
                    # Hence the +1
                    index_out.append(j + (i + 1) * self._num_state)
        else:
            # else, happy times!
            for i in index_list:
                index_out.append(state_index + (i + 1) * self._num_state)

        return np.sort(np.array(index_out)).tolist()

    def _getTargetParamIndex(self):
        """
        Get the indices of the targeted parameters
        """
        # we assume that all the parameters are targets
        if self._targetParam is None:
            index_list = range(0, self._num_param)
        else:
            # only select from the list
            index_list = list()
            # note that "i" is a string here
            for i in self._targetParam:
                index_list.append(self._ode.get_param_index(i))

        return index_list

    def _getTargetStateSensIndex(self):
        # as usual, locate the index of the state
        state_index = self._ode.get_state_index(self._stateName)

        # build the indexes to locate the correct parameters
        index_out = list()
        # locate the target indexes
        index_list = self._getTargetStateIndex()

        ## Note to self. We do not use list comprehension here because it will
        ## exceed the 80 character limit
        n_s = self._num_state
        n_p = self._num_param
        if isinstance(state_index, list):
            for j in state_index:
                for i in index_list:
                    # always ignore the first numState because they are outputs
                    # from the actual ode and not the sensitivities
                    index_out.append(j + (i + 1 + n_p)*n_s)
        else:
            # else, happy times!
            for i in index_list:
                index_out.append(state_index + (i + 1 + n_p)*n_s)

        return np.sort(np.array(index_out)).tolist()

    def _getTargetStateIndex(self):
        """
        Get the indices of our targeted states
        """
        if self._targetState is None:
            index_list = range(self._num_state)
        else:
            index_list = [self._ode.get_state_index(i) for i in self._targetState]

        return index_list

    def _setParamInput(self, theta):
        if self._targetParam is None:
            if len(theta) != self._num_param:
                raise InputError("Expecting input to all the parameters")
            else: # happy, standard case
                self._setParam(theta)
        else:
            if len(theta) == len(self._targetParam):
                self._unrollParam(theta)
            else:
                raise InputError("Expecting input theta to be of length " +
                                 str(len(self._targetParam)))

    def _setParamStateInput(self, theta):
        """
        Set both the parameters and initial condition :math:`x_{0}`
        """
        if self._targetParam is None and self._targetState is None:
            # we are expecting the standard case here
            if len(theta) != (self._num_state + self._num_param):
                raise InputError("Expecting a guess of the initial value, " +
                                 "use diff_loss() " +
                                 "instead for just parameter estimation")
            else:
                self._setX0(theta[-self._num_state:])
                self._setParam(theta[:self._num_param])
        else:
            if self._targetParam is None:
                # this mean all the parameters or without the parameters
                if len(theta) == len(self._targetState):
                    # without parameters
                    self._unrollState(theta)
                elif len(theta) == (self._num_param + len(self._targetState)):
                    # the parameters first
                    self._setParam(theta[:self._num_param])
                    # then the states
                    # x0 = theta[-len(self._targetState):]
                    self._unrollState(theta[-len(self._targetState):])
                else:
                    raise InputError("Expecting input to all the parameters " +
                                     "and to the states with length %s" %
                                     len(self._targetState))
            elif self._targetState is None:
                # this mean all the state or without the states
                if len(theta) == self._num_param:
                    # without the states, obviously using the wrong function
                    # call
                    raise InputError("Input has the same length as the " +
                                     "number of parameters. If the initial " +
                                     "conditions for the states are not " +
                                     "required, use diff_loss() instead")
                elif len(theta) == (self._num_state + self._num_param):
                    # all the states
                    # begin setting the information
                    self._setParam(theta[:self._num_param])
                    # then the states
                    # x0 = theta[-self._num_state:]
                    self._setX0(theta[-self._num_state:])
                elif len(theta) == (self._num_state + len(self._targetParam)):
                    # again we have all the states
                    self._unrollParam(theta[:len(self._targetParam)])
                    # x0 = theta[-self._num_state:]
                    self._setX0(theta[-self._num_state:])
                else: # happy
                    raise InputError("The number of input is just plain " +
                                     "wrong. Cannot help you further.")
            else:
                # we have both set of input
                l1, l2 = len(self._targetParam), len(self._targetState)
                if len(theta) == (l1 + l2):
                    # WOOT "reached correct place"
                    x0 = theta[-l2:]
                    theta = theta[:l1]
                    self._unrollState(x0)
                    self._unrollParam(theta)
                else:
                    raise InputError("Input of length " + str(len(theta)) +
                                     ": Expecting input to the parameters " +
                                     "of length " + str(l1) +
                                     " and to the states of length " + str(l2))

    def _setParam(self, theta):
        """
        Set the parameters
        """
        if self._num_param == 0:
            self._theta = None
        else:
            if self._targetParam is not None:
                theta = ode_utils.check_array_type(theta)
                thetaDict = dict()

                l1, l2 = len(theta), len(self._targetParam)
                if len(self._targetParam) > 1:
                    if len(theta) != len(self._targetParam):
                        raise InputError("Input length = %s but we expect %s" %\
                                         (l1, l2))
                    # begin to construct our dictionary
                    for i in range(l1):
                        thetaDict[self._targetParam[i]] = theta[i]
                else:
                    if isinstance(theta, Number):
                        thetaDict[self._targetParam[0]] = theta
                    elif len(theta) > 1:
                        raise InputError("Input length = "  +str(l1) +
                                         " but we only have one parameter")
                    else:
                        if isinstance(self._targetParam[0], ODEVariable):
                            thetaDict[str(self._targetParam[0])] = theta[0]
                        else:
                            thetaDict[self._targetParam[0]] = theta[0]
                self._theta = thetaDict
            else:
                # conver to something sensible
                theta = ode_utils.check_array_type(theta)
                self._theta = np.copy(theta)

    def _setWeight(self, n, p, w):
        # note that we NEVER scale the weights
        # also note that we can use the weights as a control
        # with normalized input

        w = ode_utils.check_array_type(w)
        if len(w) == w.size:
            m, q = len(w), 1
        else:
            m, q = w.shape

        if p == q:
            if n == m:
                self._stateWeight = w
            elif m == 1:
                self._stateWeight = np.ones((n, p))*w
            else:
                raise InputError("Number of input weights is not equal " +
                                "to the number of observations")
        elif p == m:
            if q == 1:
                self._stateWeight = np.ones((n, p))*w
            else:
                raise InputError("Number of input weights is not equal " +
                                 "to number of states")
        else:
            if q == 1 and m == 1:
                self._stateWeight = np.ones((n, p))*w
            else:
                raise InputError("Number of input weights differs from " +
                                 "the number of observations")

    def _setX0(self, x0):
        """
        Set the initial value, pretty much only used when we are
        dealing with estimating the initial value as well
        """
        x0 = ode_utils.check_array_type(x0)
        self._x0 = np.copy(x0)

    def _setLossType(self):
        """
        we set the loss type desired. This is the method that will
        be override in the module odeLoss.  Basically, all other
        operations remains but this will change.
        """
        self._lossObj = Square(self._y, self._stateWeight)
        return self._lossObj

    def _unrollParam(self, theta):
        """
        The difference between this and _setParam is that this method
        only works if the self._theta exist, i.e. _setParam has been
        invoked previously
        """
        if self._targetParam is not None:
            # both are dictionary, straight copy over
            if isinstance(theta, dict):
                for k, v in theta.items():
                    self._theta[k] = v
            else:
                # theta only contains the value
                for i, ti in enumerate(theta):
                    # unroll the name of the parameters
                    param_str = self._targetParam[i]
                    self._theta[param_str] = ti
        else: # it is none, we swap all the values
            if isinstance(self._theta, dict):
                i = 0
                for k, v in self._theta.items():
                    self._theta[k] = theta[i]
                    i += 1
            else:
                for i in range(len(theta)):
                    self._theta[i] = theta[i]

    def _unrollState(self, x0):
        """
        If the target state are not entered in sequence, then we need
        to adjust and assign the correct index
        """
        for i, s in enumerate(self._targetState):
            index = self._ode.get_state_index(s)
            self._x0[index] = x0[i]

    def thetaCallBack(self, x):
        """
        Print x, the parameters
        """
        print(x)

    def thetaCallBack2(self, x, f):
        """
        Print x and f where x is the parameter of interest
        and f is the objective function

        Parameters
        ----------
        x:
            parameters
        f:
            f(x)
        """
        print("f(x) = " + str(f) + " ; x = " + str(x))
