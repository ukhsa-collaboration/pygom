# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:32:31 2020

@author: Jonathan.Carruthers
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot

""" v7: - allowing us to specify a constraint on initial conditions when inferring initial conditions """

# could be included within utilR/distn.py
##### multivariate normal distribution
def dmvnorm(x, mean=None, sigma=None):
    """
    See
    https://www.rdocumentation.org/packages/mvtnorm/versions/1.0-12/topics/mvnorm
    """
    if mean is None:
        mean = np.repeat(0, len(x))
    if sigma is None:
        sigma = np.identity(len(x))
    return st.multivariate_normal.pdf(x, mean=mean, cov=sigma)

def rmvnorm(n, mean, sigma, seed=None):
    '''
    See
    https://www.rdocumentation.org/packages/mvtnorm/versions/1.0-12/topics/mvnorm
    '''
    return st.multivariate_normal.rvs(mean=mean, cov=sigma, size=n)


def get_tolerance(self,g,dist):
    # choose the tolerance given the generation number and how q and tol are defined
    if g == 0:
        if not hasattr(self.tol, "__len__"):
            return self.tol
        else:
            return self.tol[0]
    else:
        if self.q is not None:
            return np.quantile(dist,self.q)
        else:
            return self.tol[g]
        
        
def sigma_nearest_neighbours(self,xx,index):
    """
    Parameters
    ----------
    self: object of class ABC
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

def get_length(attr):
    if hasattr(attr,"__len__"):
        return len(attr)
    else:
        return 0
    
def get_state_labels(attr):
    if hasattr(attr,"__len__"):
        return [name+'(0)' for name in attr]
    else:
        return []
    
class ABC():
    """
    Parameters
    ----------
    object: object of class `BaseLoss` e.g. SquareLoss
    bounds: array like of tuple
            specifies the bounds for each parameter
    log10scale: array like of bool
            specifies which parameters should be inferred on a log10 scale
    constraint: tuple
            specifies the total population size and which state's initial 
            condition should be changed to conserve the population size.
    """
    def __init__(self, object, bounds, log10scale=None, constraint=None):
        self.obj = object
        
        priors = zip(*bounds)
        self.prior_low, self.prior_high = priors
        self.prior_range = np.subtract(self.prior_high,self.prior_low)
        
        if self.obj._targetParam is None:
            # perform a normal inference with all of the parameters and any unknown initial conditions
            self.numParam = self.obj._num_param + get_length(self.obj._targetState)
        else:
            # perform inference on specified parameters only, along with any unknown initial conditions
            self.numParam = len(self.obj._targetParam) + get_length(self.obj._targetState)
    
        # perform some checks
        if log10scale is None:
            self.log = np.zeros(self.numParam)
        else:
            self.log = np.array(log10scale)
            
        if constraint is not None:
            self.pop_size = constraint[0]
            self.con_state = self.obj._ode.get_state_index(constraint[1])[0]
            # indices of the states that are not changed when conserving total population size
            self.con_state_indices = [i for i in range(self.obj._ode.num_state) if i!=self.con_state]
            
        assert len(bounds) == self.numParam, "A prior distribution must be specified for each parameter"
        assert len(self.log) == self.numParam, "Please specify whether or not each parameter should be inferred on a log-scale"
        
        
    def _get_update_function(self):
        if self.obj._targetState is None:
            return self.obj._setParam
        else:
            return self.obj._setParamStateInput
        
        
    def _log_parameters(self, params):
        if hasattr(params,"__len__"):
            params[self.log>0] = 10**params[self.log>0]
        else:
            if self.log[0] == 1:
                params = 10**params
        return params
            

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
        self.acceptance_rate = np.zeros(self.G)
    
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
        
        # empty array to store the values of cost
        dist = np.empty(self.N)
        
        for g in range(rerun,self.G+rerun):
            tolerance = get_tolerance(self,g-rerun,dist)
            
            i = 0
            total_counter = 0
            
            # making copies of the parameters and weights for referencing
            res_old = self.res.copy()
            w_old = self.w.copy()/sum(self.w)

            # getting the correct covariance matrix            
            if (self.M is not None):
                mnn_sigma = [sigma_nearest_neighbours(self,res_old,k) for k in range(self.N)]
            else:
                sigma = np.cov(self.res.T)
            
            while i < self.N:
                #print(i)
                total_counter += 1
                if g == 0: 
                    trial_params = st.uniform.rvs(self.prior_low,self.prior_range)  # <- note the definition of st.uniform.rvs differs from numpy
                else:
                    random_index = np.random.choice(self.N,p=w_old)
                    if self.M is not None:
                        sigma = mnn_sigma[random_index]
                    trial_params = rmvnorm(1,mean=res_old[random_index],sigma=sigma)
                w1 = np.prod(st.uniform.pdf(trial_params, self.prior_low, self.prior_range))
                if w1:
                    # formatting trial_params e.g. converting from log-scale, ensuring total population size is conserved
                    temp_trial_params = trial_params.copy()
                    temp_trial_params = self._log_parameters(temp_trial_params)
                    par_update(temp_trial_params)
                    if hasattr(self,"con_state"): # i.e. if there is a total population size that needs to be conserved...
                        # ...set the new initial condition equal to pop size minus the sum of the remaining initial conditions
                        self.obj._x0[self.con_state] = self.pop_size - self.obj._x0[self.con_state_indices].sum() 
                    #print(trial_params)
                    #print(self.obj._x0)
                    #print(self.obj._ode.parameters)
                    
                    cost = self.obj.cost()
                    if cost < tolerance:
                        self.res[i] = trial_params
                        dist[i] = cost
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
            self.next_tol = np.quantile(dist,self.q)        


    def continue_posterior_sample(self, N, tol, G=1, q=None, M=None, progress=False):
        # perform checks
        assert N == self.N, "For now, set the sample size to be the same as the previous run"
        assert hasattr(self, "res"), "Use 'get_posterior_sample' before 'continue_posterior_sample'"

        if hasattr(tol, "__len__"):
            assert tol[0] <= self.final_tol, "The initial tolerance is greater than the final tolerance from the previous run"
        else:
            assert tol <= self.final_tol, "The initial tolerance is greater than the final tolerance from the previous run"
            
        self.get_posterior_sample(N, tol, G, q, M, progress, rerun=True)
        
                 
    def plot_posterior_histograms(self,max_ncol=4):#,bin_width=0.01):
        if self.obj._targetParam is None:
            axis_labels = [str(i) for i in self.obj._ode.param_list] + get_state_labels(self.obj._targetState)
        else:
            axis_labels = self.obj._targetParam + get_state_labels(self.obj._targetState)
            
        nrows = 1 + (self.numParam-1)//max_ncol
        ncols = min(self.numParam,max_ncol)
        f, axarr = matplotlib.pyplot.subplots(nrows,ncols)

        if self.numParam == 1:
            axarr.plot([self.prior_low[0],self.prior_high[0]],2*[1/self.prior_range[0]],color='r',ls='--',alpha=0.75,lw=1.5)
            xvalues = np.linspace(self.prior_low[0],self.prior_high[0],1+int(self.prior_range[0]/0.001))
            kernel = st.gaussian_kde(self.res.T[0])
            axarr.fill_between(xvalues,kernel(xvalues),facecolor=(0,0,1,0.2),edgecolor=(0,0,1,1),lw=2.0)
            axarr.set(xlim=(self.prior_low[0],self.prior_high[0]), ylim=(0), xlabel=axis_labels[0])
            axarr.set(adjustable='box')
        else:
            pp = 0
            for pp, ax in enumerate(f.axes):
                # plotting the pdf of the prior distribution
                ax.plot([self.prior_low[pp],self.prior_high[pp]],2*[1/self.prior_range[pp]],color='r',ls='--',alpha=0.75,lw=1.5)
                
                # plotting a 'traditional' histogram
                #bins = np.linspace(self.prior_low[pp],self.prior_high[pp],1+int(self.prior_range[pp]/bin_width))
                #hist = np.histogram(self.res.T[pp],bins=bins,density=True)[0]
                #ax.bar(bins[:-1],hist,align='edge',width=bin_width,edgecolor=(0,0,1,1),facecolor=(0,0,1,0.25),lw=1.5)
                
                # using kernel density estimation to plot a smoothed histogram
                xvalues = np.linspace(self.prior_low[pp],self.prior_high[pp],1+int(self.prior_range[pp]/0.001))
                kernel = st.gaussian_kde(self.res.T[pp])
                ax.fill_between(xvalues,kernel(xvalues),facecolor=(0,0,1,0.2),edgecolor=(0,0,1,1),lw=2.0)
                ax.set(xlim=(self.prior_low[pp],self.prior_high[pp]), ylim=(0), xlabel=axis_labels[pp])
                ax.set(adjustable='box')             
                if pp == self.numParam-1: break
        #f.tight_layout()
        #f.show()
        
               
    def plot_pointwise_predictions(self,new_time=None,max_ncol=3):
        par_update = self._get_update_function()
        
        # formatting the array so we don't get an error when fitting to one variable
        if len(self.obj._stateName) == 1:
            # (the following didn't work because we must have run it with new_time starting from 0 instead of just above 0. This meant the method
            # never finished and so the observeT was never reset at the end. When running it again, the wrong observeT was then being used to reshape the array.)
            #self.obj._y = self.obj._y.reshape((1,len(self.obj._observeT))).T <- not sure why this wasn't working 
            self.obj._y = self.obj._y.reshape((1,len(self.obj._y))).T
        
        data_times = self.obj._observeT.copy()
        if new_time is not None:
            self.obj._observeT = new_time
        
        # setting up an empty array to store the result for each parameter set
        tt = self.obj._observeT
        numStates = self.obj._num_state
        stateList = [str(i) for i in self.obj._ode._stateList]
        
        nrows = 1 + (numStates-1)//max_ncol
        ncols = min(numStates,max_ncol)
        
        # finding the point-wise median solution and 95% credible regions
        solution = np.zeros((self.N,len(tt)*numStates))
        for i in range(self.N):
            params = self.res[i].copy()
            params = self._log_parameters(params)
            par_update(params)
            if hasattr(self,"con_state"):
                self.obj._x0[self.con_state] = self.pop_size - self.obj._x0[self.con_state_indices].sum()
            solution[i] = self.obj._getSolution(all_solution=True).T.flatten() 
        
        median = np.median(solution,axis=0).reshape((numStates,len(tt)))
        credible_95_high = np.quantile(solution,q=0.975,axis=0).reshape((numStates,len(tt)))
        credible_95_low = np.quantile(solution,q=0.025,axis=0).reshape((numStates,len(tt)))
        
        self.median = median
        self.credible_95_high = credible_95_high
        self.credible_95_low = credible_95_low
        
        # plotting the solution
        f, axarr = matplotlib.pyplot.subplots(nrows,ncols,squeeze=0)
        dd = 0
        for pp, ax in enumerate(f.axes):
            ax.plot(tt,median[pp],color='r')
            ax.fill_between(tt,credible_95_low[pp],credible_95_high[pp],color='gray',alpha=0.5)
            if pp in self.obj._ode.get_state_index(self.obj._stateName):
                ax.scatter(data_times,self.obj._y[::,dd], marker='o',facecolor='gray',alpha=0.5,edgecolor=(0,0,0,1))
                dd += 1
            ax.set(xlabel='Time',title=stateList[pp])
            if pp == self.obj._num_state-1: break
        #f.tight_layout()
        #f.show()
        
        # reset the times in case we want to refine the posterior sample
        self.obj._observeT = data_times


    def plot_scatter(self,logscale=True):
        posterior_sample = self.res.copy()
        if not logscale:
            posterior_sample[:,self.log>0] = 10**posterior_sample[:,self.log>0]
            
        if self.obj._targetParam is None:
            axis_labels = [str(i) for i in self.obj._ode.param_list] + get_state_labels(self.obj._targetState)
        else:
            axis_labels = self.obj._targetParam + get_state_labels(self.obj._targetState)
        
        f, axarr = matplotlib.pyplot.subplots(self.numParam,self.numParam,squeeze=0)
        for i in range(self.numParam):
            for j in range(self.numParam):
                axarr[i,j].scatter(posterior_sample.T[j],posterior_sample.T[i],marker='o',s=5, color='b', alpha=0.2)
                axarr[i,j].set(xlabel=axis_labels[j],ylabel=axis_labels[i])
        for ax in axarr.flat:
            ax.label_outer()
        #f.tight_layout()
        #f.show()
                