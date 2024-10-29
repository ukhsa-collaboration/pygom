"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOde']
import logging

import copy
from numbers import Number

import numpy as np
import sympy
import sympy.matrices.matrices
import scipy.stats

from .deterministic import DeterministicOde
from .stochastic_simulation import cle, exact, firstReaction, tauLeap, hybrid
from .transition import TransitionType, Transition
from ._model_errors import InputError, SimulationError
from ._model_verification import checkEquation, simplifyEquation
from . import _ode_composition
from . import ode_utils


class HasNewTransition(ode_utils.CompileCanary):
    states = ['ode',
              'Jacobian',
              'diffJacobian',
              'grad',
              'GradJacobian',
              'transitionMatrixCompile',
              'transitionVector',
              'birthDeathRateCompile',
              'computeTransitionMeanVar',
              'transitionJacobian',
              "_vMatCompile",
              "_transitionVectorCompile",
              "_transitionMeanCompile",
              "_transitionVarCompile"]

class SimulateOde(DeterministicOde):
    '''
    This builds on top of :class:`DeterministicOde` which we
    simulate the outcome instead of solving it deterministically

    Parameters
    ----------
    state: list
        A list of states (string)
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string,string))
    transition: list
        A list of transition (:class:`Transition`)
    birth_death: list
        A list of birth or death process (:class:`Transition`)
    ode: list
        A list of ode (:class:`Transition`)

    '''

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 birth_death=None,
                 ode=None):
        '''
        Constructor that is built on top of DeterministicOde
        '''

        super(SimulateOde, self).__init__(state,
                                          param,
                                          derived_param,
                                          transition,
                                          birth_death,
                                          ode)

        self._hasNewTransition = HasNewTransition()

        # self.event=None
        self.pre_tau=None
        self._epsilon=0.03

        # need a manual override because it is possible that we
        # want to perform simulation in a parallel/distributed manner
        # and there are issues with pickling fortran objects
        self._SC = ode_utils.compileCode(backend='cython')


    def __repr__(self):
        return "SimulateOde" + self._get_model_str()

    def exact(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using an exact method starting from time
        t0 to t1 with the starting state values of x0

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(exact(x0, t0, t1, self._vMat, self._transitionVectorCompile,
                     output_time=output_time))

    ###########################################################################
    #
    # Solver functions
    #
    ###########################################################################

    def cle(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using the CLE approximation starting from time
        t0 to t1 with the starting state values of x0.  The CLE approximation
        is performed using a simple Euler-Maruyama method with step size h.
        We assume that the input parameter transition_func provides
        :math:`f(x,t)` while the CLE is defined as
        :math:`dx = x + V*h*f(x,t) + \\sqrt(f(x,t))*Z*\\sqrt(h)`
        with :math:`Z` being standard normal random variables.

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(cle(x0, t0, t1, self._vMat, self._transitionVectorCompile,
                   output_time=output_time))

    def hybrid(self, x0, t0, t1, output_time=False):
        '''
        Stochastic simulation using an hybrid method that uses either the
        first reaction method or the :math:`\\tau`-leap depending on the
        size of the states and transition rates.  Starting from time
        t0 to t1 with the starting state values of x0.

        Parameters
        ----------
        x: array like
            state vector
        t0: double
            start time
        t1: double
            final time
        '''
        return(hybrid(x0, t0, t1, self._vMat,
                      self._transitionVectorCompile,
                      self.transition_mean,
                      self.transition_var,
                      output_time=output_time))

    def simulate_param(self, t, iteration, parallel=False, full_output=False):
        '''
        Simulate the ode by generating new realization of the stochastic
        parameters and integrate the system deterministically.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        iteration: int
            number of iterations you wish to simulate
        parallel: bool, optional
            Defaults to True
        full_output: bool, optional
            if we want additional information, Y_all in the return,
            defaults to false

        Returns
        -------
        Y: :class:`numpy.ndarray`
            of shape (len(t), len(state)), mean of all the simulation
        Y_all: :class:`np.ndarray`
            of shape (iteration, len(t), len(state))
        '''

        # if our parameters not stochastic, then we are going to
        # throw a warning because simulating a deterministic system is
        # just plain stupid
        if self._stochasticParam is None:
            raise InputError("Deterministic parameters.")
        if iteration is None:
            raise InputError("Need to specify the number of iterations")
        if t is None:
            raise InputError("Need to specify the time we wish to observe")

        self._odeSolution = self.integrate(t)

        # try to compute the simulation in parallel
        if parallel:
            try:
                for i in self._stochasticParam:
                    if isinstance(i, scipy.stats._distn_infrastructure.rv_frozen):
                        raise Exception("Cannot perform parallel simulation "
                                        +"using a serialized object as distribution")
                # check the type of parameter we have as input
                import dask.bag
                y = list()
                for i in range(iteration):
                    y_i = list()
                    for key, rv in self._stochasticParam.items():
                        y_i += [{key:rv.rvs(1)[0]}]
                    y += [y_i]
                # y = [rv.rvs(iteration) for rv in self._stochasticParam.values()]
                # y = np.array(list(zip(*y)))
                def sim(x):
                    self.parameters = x
                    return self.integrate(t)

                # def sim(t1): return(self.integrate(t1))

                # xtmp = dask.bag.from_sequence([t]*iteration)
                xtmp = dask.bag.from_sequence(y)
                solutionList = xtmp.map(sim).compute()
            except Exception: # as e:
                # print(e)
                # print("Serial")
                solutionList = [self.integrate(t) for i in range(iteration)]
        else:
            solutionList = [self.integrate(t) for i in range(iteration)]

        # now make our 3D array
        # the first dimension is the number of iteration
        Y = np.dstack(solutionList).mean(axis=2)

        if full_output:
            return Y, solutionList
        else:
            return Y
        
    def solve_determ(self, t, iteration=None, parallel=False, full_output=False):
            '''
            Simulate the ode by generating new realization of the stochastic
            parameters and integrate the system deterministically.

            Parameters
            ----------
            t: array like
                the range of time points which we want to see the result of
            iteration: int
                number of iterations you wish to simulate
            parallel: bool, optional
                Defaults to True
            full_output: bool, optional
                if we want additional information, Y_all in the return,
                defaults to false

            Returns
            -------
            Y: :class:`numpy.ndarray`
                of shape (len(t), len(state)), mean of all the simulation
            Y_all: :class:`np.ndarray`
                of shape (iteration, len(t), len(state))
            '''
            if t is None:
                raise InputError("Need to specify the time we wish to observe")
            
            # If parameters are not random then return one integration
            if self._stochasticParam is None:
                solution = self.integrate(t)
                return solution
            
            # Otherwise, proceed for random parameters and verify expected extra parameters are present.
            if iteration is None:
                raise InputError("Need to specify the number of iterations")

            self._odeSolution = self.integrate(t)

            # try to compute the simulation in parallel
            if parallel:
                try:
                    for i in self._stochasticParam:
                        if isinstance(i, scipy.stats._distn_infrastructure.rv_frozen):
                            raise Exception("Cannot perform parallel simulation "
                                            +"using a serialized object as distribution")
                    # check the type of parameter we have as input
                    import dask.bag
                    y = list()
                    for i in range(iteration):
                        y_i = list()
                        for key, rv in self._stochasticParam.items():
                            y_i += [{key:rv.rvs(1)[0]}]
                        y += [y_i]
                    # y = [rv.rvs(iteration) for rv in self._stochasticParam.values()]
                    # y = np.array(list(zip(*y)))
                    def sim(x):
                        self.parameters = x
                        return self.integrate(t)

                    # def sim(t1): return(self.integrate(t1))

                    # xtmp = dask.bag.from_sequence([t]*iteration)
                    xtmp = dask.bag.from_sequence(y)
                    solutionList = xtmp.map(sim).compute()
                except Exception: # as e:
                    # print(e)
                    # print("Serial")
                    solutionList = [self.integrate(t) for i in range(iteration)]
            else:
                solutionList = [self.integrate(t) for i in range(iteration)]

            # now make our 3D array
            # the first dimension is the number of iteration
            Y = np.dstack(solutionList).mean(axis=2)

            if full_output:
                return Y, solutionList
            else:
                return Y

    # Same as function below, just trying out new naming convention
    def solve_stochast(self, t, iteration, parallel=False,
                       exact=False, full_output=False):
        '''
        Simulate the ode using stochastic simulation.  It switches
        between a first reaction method and a :math:`\\tau`-leap
        algorithm internally. When a parallel backend exists, then a new random
        state (seed) will be used for each processor.  This is due to a lack
        of appropriate parallel seed random number generator in python.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
            or the final time point
        iteration: int
            number of iterations you wish to simulate
        parallel: bool, optional
            Defaults to True
        exact: bool, optional
            True if exact simulation is desired, defaults to False
        full_output: bool, optional
            if we want additional information, sim_T

        Returns
        -------
        sim_X: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        sim_Jump: list of :class:`numpy.ndarray`
            Number times each transition happens per timestep
        sim_T: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        '''
        print("Setting up stochastic simulation")

        assert len(self._odeList) == 0, \
            "Currently only able to simulate when only transitions are present"
        assert np.all(np.mod(self._x0, 1) == 0), \
            "Can only simulate a jump process with integer initial values"

        # Determine if results are output to predefined timepoints or the timepoints
        # as determined by the numerical solver
        timePoint = False

        if isinstance(t, Number):#, (int, float, np.int64, np.float64)):
            finalT = t
        elif isinstance(t, (list, tuple)):
            t = np.array(t)
            if len(t) == 1:
                finalT = t
            else:
                finalT = t[-1:]
                timePoint = True
        elif isinstance(t, np.ndarray):
            finalT = t[-1:]
            timePoint = True
        else:
            raise InputError("Unknown data type for time")

        if parallel:
            try:
                import dask.bag
                logging.debug("Using Dask for parallel simulation")
                def jump_partial(final_t): return(self._jump(final_t,
                                                             exact=exact,
                                                             full_output=True,
                                                             seed=True))

                xtmp = dask.bag.from_sequence(np.ones(iteration)*finalT)
                xtmp = xtmp.map(jump_partial).compute()
            except Exception as e:
                raise e
                logging.warning("Parallel simulation failed reverting to serial")
                xtmp = [self._jump(finalT, exact=exact, full_output=True) for _i in range(iteration)]
        else:
            logging.debug("Performing serial simulation")
            xtmp = [self._jump(finalT, exact=exact, full_output=True) for _i in range(iteration)]

        # Unpack output
        xmat = list(zip(*xtmp))
        simXList, simJumpList, simTList, simdTList = list(xmat[0]), list(xmat[1]), list(xmat[2]), list(xmat[3])

        # Process simulation output if user has specified target time steps
        if timePoint:
            for _i in range(len(simXList)):
                # TODO: this seems like an overcomplicated way to do things

                # unroll, always the first element
                # it is easy to remember that we are accessing the first
                # element because pop is spelt similar to poop and we
                # all know that your execute first in first out when you
                # poop!

                # Get time points of run _i
                simT = simTList.pop(0)      # get timepoints and remove (temporarily)

                # 1. Process states
                simX = simXList.pop(0)      # get states and remove (will be appended after processing)
                if exact:
                    x = self._extractObservationAtTime(simX, simT, t)
                else:
                    x = self._interpolateObservationAtTime(simX, simT, t)
                simXList.append(x)          # processed results now go at the end of the list

                # 2. Process jumps
                simJump = simJumpList.pop(0)
                jump=self._addJumpsBetweenTime(simJump, simT, t, exact)
                simJumpList.append(jump)    # processed results go at end of list

                # # 3. Process timesteps (essentially send to back of list to match new X and jump positions)
                # dt=simdTList.pop(0)         # send dt to back too
                # simdTList.append(dt)

                # simTList.append(t)          # do same with timepoints
                #                             # TODO: not useful to have user timepoints repeatd for every iteration

        # note that we have to remain in list form because the number of
        # simulation will be different if we are not dealing with
        # a specific set of time points

        if full_output:
            if timePoint:
                return simXList, simJumpList, t
            else:
                return simXList, simJumpList, simTList
        else:
            return simXList

    def _jump(self, finalT, exact=False, full_output=True, seed=None):
        '''
        Jumps from the initial time self._t0 to the input time finalT
        '''

        if isinstance(self._stochasticParam, dict):
            self.parameters = self._stochasticParam

        # initial time
        assert self._t0 is not None, "No initial time"
        assert self._x0 is not None, "No initial state"

        t = self._t0.tolist()
        x = copy.deepcopy(self._x0)

        # holders and record information
        xList = [x.copy()]          # states
        tList = [t]                 # timepoints
        dtList=[]                   # timesteps (can be inferred from timepoints, but useful for now to debug tau leap)
        jumpList=[]                 # transitions

        # New idea: Compile whatever is necessary at this point
        if not hasattr(self, "_vMat") \
                or self._vMat is None \
                or getattr(self._hasNewTransition, "_vMatCompile"):
            self._computeStateChangeMatrix()
            self._computeReactantMatrix()                       # TODO: Not tidy
        self.compile_sympy_object("_vMat", "_vMatCompile")

        if not hasattr(self, "_transitionVector") \
                or self._transitionVector is None \
                or getattr(self._hasNewTransition, "_transitionVectorCompile"):
            self._computeTransitionVector()
        self.compile_sympy_object("_transitionVector", "_transitionVectorCompile")

        if not hasattr(self, "_transitionMean") or not hasattr(self, "_transitionVar") \
                or self._transitionMean is None or self._transitionVar is None \
                or getattr(self._hasNewTransition, "_transitionMeanCompile") \
                or getattr(self._hasNewTransition, "_transitionVarCompile"):
            self._computeTransitionMeanVar()
        self.compile_sympy_object("_transitionMean", "_transitionMeanCompile")
        self.compile_sympy_object("_transitionVar", "_transitionVarCompile")

        if not hasattr(self, "_lambdaMat") \
                or self._lambdaMat is None :
            self._computeReactantMatrix()

        # keep jumping, Whoop Whoop (put your hands up!)
        while t < finalT:
            # Take a timetep

            # # TODO: Hacky fix to do transition of given size if condition met.
            # #       has been seen to work but comment out for now to dev other more
            ##        important code
            # # Check if one-off condition met.
            # if self.event!=None:
            #     if eval(self.event.condition)==True and self.event.event_occured==False:
            #         x=self.event.action(x)

            # TODO: Are there many modelling scenarios where it's necessary for the 
            #       v matrix to change? The rates might change with time, but do
            #       the underlying effects of a transition?
            vMat=self._vMatCompile(x,t)
            lambdaMat=self._lambdaMat
            try:
                if exact:
                    # Use Gillespie algorithm for entire simulation
                    (t,
                     jump_time,
                     x,
                     jumps,
                     success) = firstReaction(x,
                                              self._state_lims,
                                              t,
                                              vMat,
                                              self._transitionVectorCompile,
                                              seed=seed)
                    if success==False:
                        break
                else:
                    #if np.min(x) < 10:
                    #Use tau leap when population of any state is small.
                    # TODO: why? If one state, eg dead, always has a population zero then we force ourselves to use Gillespie
                    (t_new,
                     jump_time,
                     x_new,
                     jumps,
                     success) = tauLeap(x,
                                        self._state_lims,
                                        t,
                                        vMat,
                                        lambdaMat,
                                        self._transitionVectorCompile,
                                        self._transitionMeanCompile,
                                        self._transitionVarCompile,
                                        self._stochasticTrans,       # For now we tell the tau leap explicitly if we take a deterministic step
                                        epsilon=self._epsilon,
                                        seed=seed,
                                        pre_tau=self.pre_tau)
                    
                    if success:
                        # Jump results in all states within their limits, continue.
                        t, x = t_new, x_new
                    else:
                        # Retry with first reaction method.
                        (t, 
                         jump_time,
                         x,
                         jumps,
                         success) = firstReaction(x,
                                                  self._state_lims,
                                                  t,
                                                  vMat,
                                                  self._transitionVectorCompile,
                                                  seed=seed)
                        
                        if success==False:
                            break

                if success:
                    xList.append(x.copy())     # TODO: why is x a copy and the rest not?
                    tList.append(t)
                    dtList.append(jump_time)
                    jumpList.append(jumps)
                else:
                    break
            except SimulationError:
                break

        return np.array(xList), np.array(jumpList), np.array(tList), np.array(dtList)

    ###########################################################################
    #
    # Functions to process simulation output
    #
    ###########################################################################

    def _extractObservationAtTime(self, X, t, targetTime):
        '''
        Given simulation and a set of time points which we would like to
        observe, we extract the observations :math:`x_{t}` with
        :math:`\\min\\{ \\abs( t - targetTime) \\}`

        Parameters
        ----------
            t (list): Timepoints of simulation
            X (list of lists): State values at each timepoint
            targetTime (list): Desired timepoints

        Returns
        -------
            X_out (np.array): Value of states at targetTime 
        '''
        X_out = []
        
        for t_target in targetTime:
            if np.any(t == t_target):
                index = np.where(t == t_target)[0][0]
            else:
                index = max(np.searchsorted(t, t_target) - 1, 0)
            X_out.append(X[index])

        return np.array(X_out)
    
    def _interpolateObservationAtTime(self, X, t, targetTime):
        '''
        Given simulation and a set of time points which we would like to
        observe, we interpolate the observations onto the desired grid.

        Parameters
        ----------
            t (list): Timepoints of simulation
            X (list of lists): State values at each timepoint
            targetTime (list): Desired timepoints

        Returns
        -------
            X_out (np.array): Value of states at targetTime 
                    
        '''

        X=np.array(X)   # Convert to numpy array so we can interpolate between timepoints

        dims=X.shape    # Get dimensions of data (timepoints x vars)
        n_state=dims[1]

        X_out=np.zeros((len(targetTime), n_state))  # empty matrix to receive scaled data = (new timepoints x vars)

        # linearly interpolate to new timepoints
        for i in range(n_state):
            X_out[:,i]=np.interp(targetTime, t, X[:,i])

        return X_out

    def _addJumpsBetweenTime(self, dX, t, targetTime, exact):
        '''
        Given number of events occuring at certain times find total
        number of occurrances between the desired timepoints.

        Parameters
        ----------
            t (list): Timepoints that events occurred
            dX (list of lists): Number of events at each timepoint
            targetTime (list): Desired timepoints
            exact (bool): If first reaction method used (otherwise tau assumed)

        Returns
        -------
            X_out (np.array): Value of states at targetTime 
                    
        '''

        dX=np.array(dX)   # convert to numpy array so we can interpolate between timepoints

        dims=dX.shape         # Get dimensions of data (timepoints x n_trans)
        n_trans=dims[1]

        # empty matrix to receive scaled data = (new timepoints x n_trans)
        # minus one because we are looking at jumps which occur betwen timepoints
        # (e.g. 2 timepoints =1 jump, 10 timepoints =9)
        X_out=np.zeros((len(targetTime)-1, n_trans))

        # if exact, each point corresponds to a transitions and has weight 1.
        for i in range(n_trans):
            if exact:
                hist, bin_edges=np.histogram(t, bins=targetTime)
            else:
                hist, bin_edges=np.histogram(t[1:], bins=targetTime, weights=dX[:,i])
            X_out[:,i]=hist            

        return X_out

    ###########################################################################
    #
    # Functions to compute sympy objects and also compile them
    #
    ###########################################################################

    def compile_sympy_object(self, obj_name, compiled_obj_name):
        '''
        Take sympy object (could be an expression or matrix of expressions)
        and compile it into a function of the systsem state and time.
        '''
        if hasattr(self, compiled_obj_name) is False \
            or getattr(self, compiled_obj_name) is None \
            or getattr(self._hasNewTransition, compiled_obj_name):

            print("... Compiling sympy object with name:", obj_name, "...", end="")
            f = self._SC.compileExprAndFormat
            if self._isDifficult:
                compile_obj=f(self._sp, getattr(self, obj_name), modules='mpmath')
            else:
                compile_obj=f(self._sp, getattr(self, obj_name))

            def eval_obj(parameters=None, time=None, state=None):
                eval_param = self._getEvalParam(state, time, parameters)
                return compile_obj(eval_param)

            def comp_obj(state, t):
                return eval_obj(time=t, state=state)

            setattr(self, compiled_obj_name, comp_obj)
            self._hasNewTransition.reset(compiled_obj_name)
            print("done")
        return None

    def _computeTransitionJacobian(self):
        '''
        Evaluate equation (7) from https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        where F_[i,j] is the change in transition rate a[i], if a transition of type j occurs:
        F_[i,j] = sum_k diff(a[i], x_k) v_[k,j]
        where k=state and v[k,j] is how much state x_k changes by if transition of type j occurs.
        '''
        if self._vMat is None:
            self._computeStateChangeMatrix()

        F = sympy.zeros(self.num_transitions, self.num_transitions)

        for i, eqn in enumerate(self._transitionVector):
            for j in range(self.num_transitions):
                for k, state in enumerate(self._iterStateList()):
                    diffEqn = sympy.diff(eqn, state, 1)                  # diff(a_i, x_k)
                    diffEqn, isDifficult = simplifyEquation(diffEqn)
                    F[i,j] += diffEqn*self._vMat[k,j]
                    self._isDifficult = self._isDifficult or isDifficult  # TODO: what is this for?

        self._transitionJacobian = F
        self._hasNewTransition.reset('transitionJacobian')

        return F

    def _computeTransitionMeanVar(self):
        '''
        This is the mean and variance of the changes in the transition rates
        (aka propensity funtions) after a potential timestep:
        equations (8a) and (8b) from https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf
        For n transitions the outputs are 2 vectors, each of length n.
        Outputs are added to self as mu and sigma2
        '''

        if hasattr(self, "_transitionJacobian") is False \
                or self._transitionJacobian is None \
                or self._hasNewTransition.transitionJacobian:
            self._computeTransitionJacobian()

        F = self._transitionJacobian

        # holders
        mu = sympy.zeros(self.num_transitions, 1)
        sigma2 = sympy.zeros(self.num_transitions, 1)

        for i, eqn_i in enumerate(self._transitionVector):
            for j, eqn_j in enumerate(self._transitionVector):
                mu[i] += F[i,j] * eqn_j
                sigma2[i] += F[i,j] * F[i,j] * eqn_j

            # If time dependence, add in another term to reflect this:
            timelike_symbols=[symb for symb in eqn_i.free_symbols if str(symb)=='t']
            is_time_dependent=len(timelike_symbols)>0
            if is_time_dependent and self.tstep:
                time_variable = [timelike_symbols][0]
                mu[i] += sympy.diff(eqn_i, time_variable, 1) # mean changes but sd does not, TODO: check this

        # add results to class
        self._transitionMean = mu
        self._transitionVar = sigma2
        self._hasNewTransition.reset('computeTransitionMeanVar')

        return None

    def total_transition(self, state, t):
        '''
        Evaluate the total transition rate given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        float
            total rate

        '''
        return sum(self._transitionVectorCompile(time=t, state=state))

    ###########################################################################
    #
    # Unrolling of ode to transitions
    #
    ###########################################################################

    def get_unrolled_obj(self):
        '''
        Returns a :class:`SimulateOde` with the same state and parameters
        as the current object but with the equations defined by a set of
        transitions and birth death process instead of say, odes
        '''
        transition = self.get_transitions_from_ode()
        bdList = self.get_bd_from_ode()

        return SimulateOde(
                           [str(s) for s in self._stateList],
                           [str(p) for p in self._paramList],
                           derived_param=self._derivedParamEqn,
                           transition=transition,
                           birth_death=bdList
                           )

    def get_transitions_from_ode(self):
        '''
        Returns a list of :class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.T
        '''
        M = self._generateTransitionMatrix()

        transition = list()
        for i, s1 in enumerate(self._stateList):
            for j, s2 in enumerate(self._stateList):
                if M[i,j] != 0:
                    t = Transition(origin=str(s1),
                                   destination=str(s2),
                                   equation=str(M[i,j]),
                                   transition_type=TransitionType.T)
                    transition.append(t)

        return transition

    def _get_A(self, A=None):
        if A is None:
            if not ode_utils.none_or_empty_list(self._odeList):
                eqn_list = [t.equation for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqn_list,
                                               *self._getListOfVariablesDict(),
                                               subs_derived=False))
                return A
            else:
                raise Exception("Object was not initialized using a set of ode")
        else:
            return A

    def get_bd_from_ode(self, A=None):
        '''
        Returns a list of:class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.B or
        TransitionType.D
        '''

        A=self._get_A(A)

        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        if len(bdList) > 0:
            M = self._generateTransitionMatrix(A)

            A1 = _ode_composition.pureTransitionToOde(M)
            diffA = sympy.simplify(A - A1)

            # get our birth and death process
            bdUnroll = list()
            states = [str(i) for i in self.state_list]

            for i, a in enumerate(diffA):
                for b in bdList:
                    if _ode_composition._hasExpression(a, b):
                        if sympy.Integer(-1) in _ode_composition.getLeafs(b):
                            bdUnroll.append(Transition(origin=states[i],
                                            equation=str(b*-1),
                                            transition_type=TransitionType.D))
                        else:
                            bdUnroll.append(Transition(origin=states[i],
                                            equation=str(b),
                                            transition_type=TransitionType.B))
                        a -= b

            return bdUnroll
        else:
            return []

    def _generateTransitionMatrix(self, A=None):#, transitionExpressionList=None):
        '''
        Finds the transition matrix from the set of ode.  It is
        important to note that although some of the functions used
        in this method appear to be the same as _getReactantMatrix
        and _getStateChangeMatrix, they are different in the sense
        that the functions called here is focused on the terms of
        the equation rather than the states.
        '''
        A=self._get_A(A)
        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        fx = _ode_composition.stripBDFromOde(A, bdList)
        states = [s for s in self._iterStateList()]
        M, _remain = _ode_composition.odeToPureTransition(fx, states, True)
        return M

    def plot(self, sim_X=None, sim_T=None):
        '''
        Plot the results of a simulation

        Takes the output of a function like `simulate_jump`

        Parameters
        ----------
        sim_X: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        sim_T: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        Notes
        -----
        If either sim_X or sim_T are None the this function will attempt to
        plot the deterministic ODE

        If we have 3 states or more, it will always be arrange such
        that it has 3 columns.  Uses the operation from
        :mod:`odeutils`
        '''
        if (sim_X is None) or (sim_T is None):
            return super(SimulateOde, self).plot()
        ode_utils.plot_stoc(sim_X, sim_T, self)
