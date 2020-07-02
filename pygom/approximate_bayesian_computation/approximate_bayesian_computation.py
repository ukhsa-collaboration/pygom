# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:32:31 2020

@author: Jonathan.Carruthers
"""
import logging

import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot
from inspect import signature

from pygom.utilR import dmvnorm, rmvnorm
from pygom.loss.ode_loss import SquareLoss, NormalLoss, PoissonLoss

""" v7: - allowing us to specify a constraint on initial conditions when inferring initial conditions """
""" v8: 15-04-2020
    - added a parameter class that allows use to more easily implement different
      prior distributions. The prior distributions are those defined in pygom.utilR:
          - normal
          - gamma
          - beta
          - uniform
    - improved plotting that makes it easy to plot pointwise predictions for
      selected states and plot posterior histograms/pairs plots for specific
      parameters.
"""

#%%
def _log_limits(par, logscale):
    # used in plot_scatter to convert axis limits from a log-scale if necessary
    if (not logscale) and (par.logscale):
       return (10**par.prior_low, 10**par.prior_high)
    else:
       return par.prior_low, par.prior_high

def get_length(attr):
    if hasattr(attr,"__len__"):
        return len(attr)
    else:
        return 0
        
def _get_target(parameters,target):
    # used to separate target_param and target_state from a single list of parameters  
    target_list = [param.name for param in parameters if param.name in target]
    if len(target_list) == 0:
        return None
    else:
        return target_list
    
def get_function(str): 
    # gets a function from a string for a distribution - is there a better way to write this?
    """
    Parameters
    ----------
    str: name of the chosen prior distribution
    """
    #try:
    #    return getattr(sys.modules["__main__"],str)
    #except AttributeError:
    #    pass
    try:
        return getattr(sys.modules["pygom.utilR"],str)
    except:
        raise AttributeError("The chosen distribution is not available, please choose a different one")

def _get_sigma(i,res,weights,indices):
        diff = res[indices] - res[i]
        return np.einsum('ij,ik,i->jk', diff, diff, weights)

#%%
""" Parameter class"""
class Parameter():
    """
    Parameters
    ----------
    name: string specifying the name of the parameter
    distname: string specifying the prior distribution
    distpars: parameters of the chosen prior distribution
    logscale: bool specifying whether parameter should be sampled on a log scale
    """
    def __init__(self, name, distname, *distpars, logscale):
        self.name = name
        self.prior_distribution = distname
        self.prior_rvs = get_function("r"+distname)
        self.prior_density = get_function("d"+distname)
        self.prior_quantile = get_function("q"+distname)
        self.prior_pars = distpars
        self.logscale = logscale

        if self.prior_distribution == "unif":
            self.prior_low, self.prior_high = self.prior_pars
        elif self.prior_distribution == "beta":
            self.prior_low, self.prior_high = 0, 1
        else:
            # we need to round these values so they're a bit nicer
            self.prior_low = np.floor(self.prior_quantile(0.001,*(self.prior_pars)))
            self.prior_high = np.ceil(self.prior_quantile(0.999,*(self.prior_pars)))
        
        sig = signature(self.prior_rvs)
        assert len([param for param in sig.parameters if param not in ["n","seed"]]) == len(distpars), "Check that the correct " \
                + "number of parameters has been specified for the chosen prior distribution."
                
    def random_sample(self):
        return self.prior_rvs(1, *(self.prior_pars), seed=None)
        
    def density(self,x):
        return self.prior_density(x, *(self.prior_pars), log=False)
    
    def plot_prior(self):
        xvalues = np.linspace(self.prior_low, self.prior_high, 1+int((self.prior_high-self.prior_low)/0.001))
        fig, axarr = matplotlib.pyplot.subplots()
        axarr.plot(xvalues,self.density(xvalues))
        axarr.set(xlim=(self.prior_low,self.prior_high), ylim=(0), xlabel=self.name, ylabel="Density")
        fig.show()  
    
    
""" create a loss object by making use of a list of parameters. 
    - This avoids the need to specify `target_param` and `target_state` since 
      these will be determined from the parameters that are provided. 
    - For ABC, it is not necessary to specify an initial guess for each parameter,
      theta, since the first generation will repeatedly sample from the prior
      distributions instead. This therefor avoids the need to specify theta. """
      
def create_loss(loss_type, parameters, ode, x0, t0, t, y, state_name,
                state_weight=None, sigma=None):
    """
    Parameters (see also class `BaseLoss`)
    ----------
    loss_type: class `BaseLoss`
    parameters: list
        a list of objects of class `Parameter`
    ode: class `DeterministicOde`
        the ode class in this package
    x0: numeric
        initial time
    t0: numeric
        initial value
    t: array like
        time points where observations were made
    y: array like
        observations
    state_name: str or list
        the state(s) which the observations came from
    state_weight: array like
        weight for the observations
    sigma: 
    """
    assert t0 != t[0], "Make sure that the times, t, do not include t0"
    assert all(param.name in (ode.param_list+ode.state_list) for param in parameters), "Parameters have been provided that are not in the model"
    
    target_param = _get_target(parameters, ode.param_list)
    target_state = _get_target(parameters, ode.state_list)
    theta = [param.random_sample() for param in parameters if param.name in target_param]
    
    if loss_type == SquareLoss:
        return SquareLoss(theta, ode, x0, t0, t, y, state_name, state_weight, target_param, target_state)
    
    elif loss_type == NormalLoss:
        return NormalLoss(theta, ode, x0, t0, t, y, state_name, sigma, target_param, target_state)
    
    elif loss_type == PoissonLoss:
        return PoissonLoss(theta, ode, x0, t0, t, y, state_name, target_param, target_state)    

#%% 
""" ABC class and methods for obtaining an approximate posterior sample/plotting the results """
class ABC():
    """
    Parameters
    ----------
    loss_object: class `BaseLoss` e.g. SquareLoss
    parameters: list
        a list of objects of class `Parameter`
    constraint: tuple
            specifies the total population size and which state's initial 
            condition should be changed to conserve the population size.
    """
    def __init__(self, loss_object, parameters, constraint=None):
        self.obj = loss_object
        self.parameters = parameters
        
        if self.obj._targetParam is None:
            # perform a normal inference with all of the parameters and any unknown initial conditions
            self.numParam = self.obj._num_param + get_length(self.obj._targetState)
        else:
            # perform inference on specified parameters only, along with any unknown initial conditions
            self.numParam = len(self.obj._targetParam) + get_length(self.obj._targetState)
    
        self.log = np.array([param.logscale for param in self.parameters])
        self.prior_range = np.array([(param.prior_high-param.prior_low) for param in self.parameters])
            
        if constraint is not None:
            self.pop_size = constraint[0]
            self.con_state = self.obj._ode.get_state_index(constraint[1])[0]
            # indices of the states that are not changed when conserving total population size
            self.con_state_indices = [i for i in range(self.obj._ode.num_state) if i!=self.con_state]
                    
        assert all(isinstance(param,Parameter) for param in parameters), "Use the Parameter class to define all parameters"
        

    def get_posterior_sample_original(self, N, tol, G=1, q=None, M=None, progress=False, rerun=False):
        """
        Parameters
        ----------
        N: integer
            the number of samples in each generation
        tol: float or array like
            the initial tolerance or sequence of decreasing tolerances
        G: integer
            the number of generations used in ABC SMC/ ABC SMC MNN
        q: float (0 < q < 1)
            the quantile used to specify the tolerance for future generations in ABC SMC/ ABC SMC MNN
        M: integer
            the number of nearest neighbours used in ABC SMC MNN (M < N)
        progress: bool
            if True, reports the generation number, acceptance rate and threshold after each generation
        rerun: bool
            if False, this is the first attempt to obtain the posterior sample
        """      
        self.N = N
        self.tol = tol
        self.G = G
        self.q = q
        self.M = M
                       
        if not rerun:
            self.res = np.zeros((self.N,self.numParam))
            self.w = np.ones(self.N)
            self.dist = np.zeros(self.N)
        self.acceptance_rate = np.zeros(self.G)
        self.tolerances = np.zeros(self.G)
        
        # perform some checks
        if self.G == 1:
            assert not hasattr(self.tol, "__len__"), "When performing rejection sampling ABC, only provide a single tolerance"
        elif self.q is None:
            assert hasattr(self.tol, "__len__"), "When performing ABC SMC, a list of tolerances or quantile must be provided"
            assert len(self.tol) == self.G, "The number of tolerances specified must be equal to the number of generations"
        else:
            assert not hasattr(self.tol, "__len__"), "When specifying a quantile, only provide an initial tolerance"
            
        if self.M is not None:
            assert (self.M < self.N), "The number of nearest neighbours must be less than the sample size (M < N). Omitting M is equivalent to M = N."
                
        # setting the appropriate function for updating the parameters/initial conditions
        par_update = self._get_update_function()
                
        for g in range(rerun, self.G+rerun):
            tolerance = self.get_tolerance(g-rerun)
            self.tolerances[g-rerun] = tolerance
            
            i = 0
            total_counter = 0
            
            # making copies of the parameters and weights for referencing
            res_old = self.res.copy()
            w_old = self.w.copy()/sum(self.w)

            # getting the correct covariance matrix  
            if (self.M is not None):
                sigma_list = [self.sigma_nearest_neighbours(res_old,k) for k in range(self.N)]
            else:
                tilde_indices = np.where(self.dist < tolerance)[0] # (this should have length self.q*self.N)
                # using einsum
                w_tilde = w_old[tilde_indices]
                w_tilde_norm = w_tilde/sum(w_tilde)
                sigma_list = [_get_sigma(i,res_old,w_tilde_norm,tilde_indices) for i in range(self.N)]
                
            while i < self.N:
                total_counter += 1
                if g == 0: 
                    trial_params = np.array([param.random_sample() for param in self.parameters])
                else:
                    random_index = np.random.choice(self.N, p=w_old)
                    sigma = sigma_list[random_index]
                    trial_params = np.atleast_1d(rmvnorm(1,mean=res_old[random_index],sigma=sigma))
                w1 = np.prod([self.parameters[i].density(trial_params[i]) for i in range(self.numParam)])
                if w1:
                    # converting from log-scale and ensuring total population size is conserved
                    model_params = self._log_parameters(trial_params.copy())
                    par_update(model_params)
                    if hasattr(self,"con_state"):
                        self.obj._x0[self.con_state] = self.pop_size - self.obj._x0[self.con_state_indices].sum() 
                    
                    cost = self.obj.cost()
                    if cost < tolerance:
                        self.res[i] = trial_params
                        self.dist[i] = cost
                        if g == 0:
                            w2 = 1
                        else:
                            # the following definition of wk is fine if the kernel is symmetric e.g. for a normal pdf we have (x-mu)**2 = (mu-x)**2
                            wk = dmvnorm(res_old, mean=self.res[i], sigma=sigma)  
                            w2 = np.dot(wk, w_old)
                        self.w[i] = w1/w2
                        i += 1
            accept_rate = 100*self.N/total_counter
            self.acceptance_rate[g-rerun] = accept_rate
            if progress: print("Generation %s \n tolerance = %.5f \n acceptance rate = %.2f%%\n" % (g+1-rerun,tolerance,accept_rate))
        self.final_tol = tolerance
        if q is not None:
            self.next_tol = np.quantile(self.dist,self.q)        
        

    def get_posterior_sample(self, N, tol, G=1, q=None, M=None, progress=False, rerun=False):
        """
        Parameters
        ----------
        N: integer
            the number of samples in each generation
        tol: float or array like
            the initial tolerance or sequence of decreasing tolerances
        G: integer
            the number of generations used in ABC SMC/ ABC SMC MNN
        q: float (0 < q < 1)
            the quantile used to specify the tolerance for future generations in ABC SMC/ ABC SMC MNN
        M: integer
            the number of nearest neighbours used in ABC SMC MNN (M < N)
        progress: bool
            if True, reports the generation number, acceptance rate and threshold after each generation
        rerun: bool
            if False, this is the first attempt to obtain the posterior sample
        """      
        self.N = N
        self.tol = tol
        self.G = G
        self.q = q
        self.M = M
                       
        if not rerun:
            self.res = np.zeros((self.N,self.numParam))
            self.w = np.ones(self.N)
            self.dist = np.zeros(self.N)
        self.acceptance_rate = np.zeros(self.G)
        self.tolerances = np.zeros(self.G)
    
        if N < 100:
            logging.warn('N is low this may cause errors')#Todo: why does rmvnorm give LinAlgError with low N? better test and catch needed
        # perform some checks
        if self.G == 1:
            assert not hasattr(self.tol, "__len__"), "When performing rejection sampling ABC, only provide a single tolerance"
        elif self.q is None:
            assert hasattr(self.tol, "__len__"), "When performing ABC SMC, a list of tolerances or quantile must be provided"
            assert len(self.tol) == self.G, "The number of tolerances specified must be equal to the number of generations"
        else:
            assert not hasattr(self.tol, "__len__"), "When specifying a quantile, only provide an initial tolerance"
            
        if self.M is not None:
            assert (self.M < self.N), "The number of nearest neighbours must be less than the sample size (M < N). Omitting M is equivalent to M = N."
                
        # setting the appropriate function for updating the parameters/initial conditions
        par_update = self._get_update_function()
            
        for g in range(rerun,self.G+rerun):
            tolerance = self.get_tolerance(g-rerun)
            self.tolerances[g-rerun] = tolerance
            
            i = 0
            total_counter = 0
            
            # making copies of the parameters and weights for referencing
            res_old = self.res.copy()
            w_old = self.w.copy()/sum(self.w)
            # Todo: place these into the dask cluster

            # getting the correct covariance matrix   
            if (self.M is not None):
                sigma_list = [self.sigma_nearest_neighbours(res_old,k) for k in range(self.N)]
            else:
                tilde_indices = np.where(self.dist < tolerance)[0] # (this should have length self.q*self.N)
                # using einsum
                w_tilde = w_old[tilde_indices]
                w_tilde_norm = w_tilde/sum(w_tilde)
                sigma_list = [_get_sigma(i,res_old,w_tilde_norm,tilde_indices) for i in range(self.N)]                
                
            #total_counter = 0
            for i in range(self.N):
                (self.w[i], 
                 rejections, 
                 self.res[i], 
                 self.dist[i]) = self._perform_generation(generation=g,
                                                  sigma_list=sigma_list,
                                                  tolerance=tolerance,
                                                  par_update=par_update,
                                                  res_old=res_old,
                                                  w_old=w_old)

                total_counter += (rejections + 1)
                
            accept_rate = 100 * self.N / total_counter
            self.acceptance_rate[g-rerun] = accept_rate
            if progress: print("Generation %s \n tolerance = %.5f \n acceptance rate = %.2f%%\n" % (g+1-rerun,tolerance,accept_rate))
        
        self.final_tol = tolerance
        if q is not None:
            self.next_tol = np.quantile(self.dist,self.q)        

    def _perform_generation(self, 
                            generation, 
                            sigma_list,
                            tolerance, 
                            par_update,
                            res_old,
                            w_old):
        '''
        Carry out a single generation
        
        Parameters
        ----------
        generation: The generation number
        mnn_sigma: covariance of the M nearest neighbours 
        tolerance: Calculated tolerence for this generation
        par_update: The update function
        res_old: Previous generation parameters
        w_old: Previous generation weights
        '''
        rejections = 0
        while True: # Todo: should be some timeout on this
            if generation == 0: 
                trial_params = np.array([param.random_sample() for param in self.parameters])
            else:
                random_index = np.random.choice(self.N,p=w_old)
                sigma = sigma_list[random_index]
                trial_params = np.atleast_1d(rmvnorm(1,
                                                     mean=res_old[random_index],
                                                     sigma=sigma))
            w1 = np.prod([self.parameters[i].density(trial_params[i]) for i in range(self.numParam)])
            if w1:
                # converting from log-scale and ensuring total population size is conserved
                model_params = self._log_parameters(trial_params.copy())
                par_update(model_params)
                if hasattr(self,"con_state"): 
                    self.obj._x0[self.con_state] = self.pop_size - self.obj._x0[self.con_state_indices].sum() 
                
                cost = self.obj.cost()
                if cost < tolerance:
                    if generation == 0:
                        w2 = 1
                    else:
                        # the following definition of wk is fine if the kernel is symmetric e.g. for a normal pdf we have (x-mu)**2 = (mu-x)**2
                        wk = dmvnorm(res_old, mean=trial_params, sigma=sigma)  
                        w2 = np.dot(wk, w_old)
                    break # sucess so escape from the while
            rejections += 1
        return (w1/w2, rejections, trial_params, cost)


    def continue_posterior_sample(self, N, tol, G=1, q=None, M=None, progress=False):
        """
        Parameters (same as get_posterior_sample)
        ----------
        N: integer
            the number of samples in each generation
        tol: float or array like
            the initial tolerance or sequence of decreasing tolerances
        G: integer
            the number of generations used in ABC SMC/ ABC SMC MNN
        q: float (0 < q < 1)
            the quantile used to specify the tolerance for future generations in ABC SMC/ ABC SMC MNN
        M: integer
            the number of nearest neighbours used in ABC SMC MNN (M < N)
        progress: bool
            if True, reports the generation number, acceptance rate and threshold after each generation
        """
        # perform checks
        assert N == self.N, "For now, set the sample size to be the same as the previous run"
        assert hasattr(self, "res"), "Use 'get_posterior_sample' before 'continue_posterior_sample'"

        if hasattr(tol, "__len__"):
            assert tol[0] <= self.final_tol, "The initial tolerance is greater than the final tolerance from the previous run"
        else:
            assert tol <= self.final_tol, "The initial tolerance is greater than the final tolerance from the previous run"
            
        self.get_posterior_sample(N, tol, G, q, M, progress, rerun=True)
        
                 
    def plot_posterior_histograms(self,plot_params=None,max_ncol=4):
        """
        Parameters
        ----------
        plot_param: list
            specifies which parameters the posterior distributions are plotted for 
        max_ncol: integer
            specifies the maximum number of columns in the figure, purely to
            avoid squashed plots.
        """        
        fit_params = [p.name for p in self.parameters]
        
        if plot_params is not None:
            assert all(p in fit_params for p in plot_params), "You are trying to plot histograms of parameters not included in the inference"
            param_indices = [fit_params.index(p) for p in plot_params]
        else:
            param_indices = [i for i in range(self.numParam)]
        
        numPlotParam = len(param_indices)
        nrows = 1 + (numPlotParam-1)//max_ncol
        ncols = min(numPlotParam,max_ncol)
        f, axarr = matplotlib.pyplot.subplots(nrows,ncols)

        if numPlotParam == 1:
            # plotting the pdf of the prior distribution
            ind = param_indices[0]
            plot_low= min(self.parameters[ind].prior_low, np.floor(min(self.res.T[ind])))
            plot_high = max(self.parameters[ind].prior_high, np.ceil(max(self.res.T[ind])))
            xvalues = np.linspace(plot_low, plot_high, 1+int((plot_high-plot_low)/0.001))
            axarr.plot(xvalues,self.parameters[ind].density(xvalues),color="r",ls="--",alpha=0.75,lw=1.5)
            
            # using kernel density estimation to plot a smoothed histogram
            kernel = st.gaussian_kde(self.res.T[ind])
            axarr.fill_between(xvalues,kernel(xvalues),facecolor=(0,0,1,0.2),edgecolor=(0,0,1,1),lw=2.0)
            axarr.set(xlim=(plot_low,plot_high), ylim=(0), xlabel=fit_params[ind])
            axarr.set(adjustable='box')
        else:
            pp = 0
            for pp, ax in enumerate(f.axes):
                # plotting the pdf of the prior distribution
                ind = param_indices[pp]
                plot_low= min(self.parameters[ind].prior_low, np.floor(min(self.res.T[ind])))
                plot_high = max(self.parameters[ind].prior_high, np.ceil(max(self.res.T[ind])))
                xvalues = np.linspace(plot_low, plot_high, 1+int((plot_high-plot_low)/0.001))
                ax.plot(xvalues,self.parameters[ind].density(xvalues),color="r",ls="--",alpha=0.75,lw=1.5)

                # using kernel density estimation to plot a smoothed histogram
                kernel = st.gaussian_kde(self.res.T[ind])
                ax.fill_between(xvalues,kernel(xvalues),facecolor=(0,0,1,0.2),edgecolor=(0,0,1,1),lw=2.0)
                ax.set(xlim=(plot_low,plot_high), ylim=(0), xlabel=fit_params[ind])
                ax.set(adjustable='box')
                if pp == numPlotParam-1: break
        f.tight_layout()
        f.show()      
        return f
                
                
    def plot_pointwise_predictions(self,plot_states=None,new_time=None,max_ncol=3):
        """
        Parameters
        ----------
        plot_states: list
            specifies which states the solution is plotted for
        new_time: array like
            an array of new times to plot the model solution for. This temporarily
            overwrites the array of times used for the inference, but this change
            is reverted once the plotting is completed in case the user wants to
            run 'continue_posterior_sample'.
        max_ncol: integer
            specifies the maximum number of columns in the figure, purely to
            avoid squashed plots.
        """
        par_update = self._get_update_function()
        
        # getting the indices of the states we want to plot the solution for
        if plot_states is not None:
            assert all(s in self.obj._ode.state_list for s in plot_states), "It is only possible to plot the solution for states already in the model"
            state_indices = [self.obj._ode.state_list.index(s) for s in plot_states]
        else:
            state_indices = [i for i in range(self.obj._num_state)]       
        
        # setting up an empty array to store the result for each parameter set
        if new_time is None:
            tt = self.obj._observeT
        else:
            tt = new_time       
        
        # formatting the array so we don't get an error when fitting to one variable
        if len(self.obj._stateName) == 1:
            self.obj._y = self.obj._y.reshape((1,len(self.obj._y))).T
                
        numStates = self.obj._num_state
        numPlotStates = len(state_indices)
        
        nrows = 1 + (numPlotStates-1)//max_ncol
        ncols = min(numPlotStates,max_ncol)
        
        # finding the point-wise median solution and 95% credible regions
        solution = np.zeros((self.N,len(tt)*numStates))
        for i in range(self.N):
            params = self.res[i].copy()
            params = self._log_parameters(params)
            par_update(params)
            if hasattr(self,"con_state"):
                self.obj._x0[self.con_state] = self.pop_size - self.obj._x0[self.con_state_indices].sum()
            self.obj._ode.parameters = self.obj._theta
            self.obj._ode.initial_state = self.obj._x0
            solution[i] = self.obj._ode.integrate(tt)[1:,].T.flatten()
        
        median = np.median(solution,axis=0).reshape((numStates,len(tt)))
        credible_95_high = np.quantile(solution,q=0.975,axis=0).reshape((numStates,len(tt)))
        credible_95_low = np.quantile(solution,q=0.025,axis=0).reshape((numStates,len(tt)))
        
        self.median = median
        self.credible_95_high = credible_95_high
        self.credible_95_low = credible_95_low
        
        # plotting the solution
        f, axarr = matplotlib.pyplot.subplots(nrows,ncols,squeeze=0)
        for pp, ax in enumerate(f.axes):
            ind = state_indices[pp]            
            ax.plot(tt,median[ind],color='r')
            ax.fill_between(tt,credible_95_low[ind],credible_95_high[ind],color='gray',alpha=0.5)
            try:
                dd = self.obj._stateName.index(self.obj._ode.state_list[ind])
                ax.scatter(self.obj._observeT,self.obj._y[::,dd], marker='o',facecolor='gray',alpha=0.5,edgecolor=(0,0,0,1))
            except:
                pass
            ax.set(xlabel='Time',title=str(self.obj._ode.state_list[ind]))
            if pp == numPlotStates-1: break
        f.tight_layout()
        f.show()
        return f


    def plot_scatter(self,plot_params=None,logscale=True):
        """
        Parameters
        ----------
        plot_params: list
            specifies which parameters should be included in the pairs plot, the list should
            ideally contain the names of two or more parameters.
        logscale: bool
            indicates whether parameters sampled on a log-scale should remain on a log-scale 
        """
        fit_params = [p.name for p in self.parameters]
        
        if plot_params is not None:
            assert all(p in fit_params for p in plot_params), "You are trying to plot parameters that were not included in the inference"
            param_indices = [fit_params.index(p) for p in plot_params]
        else:
            param_indices = [i for i in range(self.numParam)]
        numPlotParam = len(param_indices)
        
        posterior_sample = self.res.copy()
        if not logscale:
            posterior_sample[:,self.log] = 10**posterior_sample[:,self.log]
                    
        f, axarr = matplotlib.pyplot.subplots(numPlotParam,numPlotParam,squeeze=0)
        for i in range(numPlotParam):
            for j in range(numPlotParam):
                iind, jind = param_indices[i], param_indices[j]
                axarr[i,j].scatter(posterior_sample.T[jind],posterior_sample.T[iind],marker='o', s=5, color='b', alpha=0.2)
                axarr[i,j].set(xlabel=fit_params[jind],ylabel=fit_params[iind])               
                axarr[i,j].set(xlim=_log_limits(self.parameters[jind],logscale))
                axarr[i,j].set(ylim=_log_limits(self.parameters[iind],logscale))
        for ax in axarr.flat:
            ax.label_outer()
        f.tight_layout()
        f.show()
     
        
    def get_tolerance(self,g):
        """
        Parameters
        ----------
        g: integer
           generation number of the ABC-SMC/MNN algorithm
        """
        # choose the tolerance given the generation number and how q and tol are defined
        if g == 0:
            if not hasattr(self.tol, "__len__"):
                return self.tol
            else:
                return self.tol[0]
        else:
            if self.q is not None:
                return np.quantile(self.dist,self.q)
            else:
                return self.tol[g]
        
        
    def sigma_nearest_neighbours(self,xx,index):
        """
        Parameters
        ----------
        xx: array like
            array of parameters
        index: integer
            index of the parameter set the nearest neighbours will be found for        
        """
        # find the covariance matrix of the M nearest particles to a specified particle
        if self.M == self.N-1:
            return np.cov(xx.T)
        else:
            diff = (xx - xx[index])/self.prior_range
            euclidean_norm = np.sum(diff**2,axis=1)
            nn = np.argpartition(euclidean_norm,self.M+1)[:self.M+1]
            return np.cov(xx[nn].T)
                
        
    def _get_update_function(self):
        if self.obj._targetState is None:
            return self.obj._setParam
        else:
            return self.obj._setParamStateInput
        
        
    def _log_parameters(self, params):
        """
        Parameters
        ----------
        params: array like
            array containing values for each of the inferred parameters
        """
        if hasattr(params,"__len__"):
            params[self.log] = 10**params[self.log]
        else:
            if self.log[0]:
                params = 10**params
        return params
    
    
    def _vprod(self,a1,a2):
        diff = (a1-a2).reshape((self.numParam,1))
        return np.dot(diff,diff.T)
    