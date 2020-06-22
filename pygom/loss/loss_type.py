"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    The different loss types.  Such as thought based on
    parametric distributions.

"""

__all__ = [
    'Square',
    'Normal',
    'Poisson',
    'Gamma',
    'NegBinom'
    ]

import numpy as np

from pygom.model._model_errors import InitializeError
from pygom.model.ode_utils import check_array_type
from pygom.utilR.distn import dnorm, dpois, gamma_mu_shape, dnbinom

class InputError(Exception):
    '''
    As the name suggest.
    '''
    pass

class baseloss_type(object):
    '''
    This baseloss_type class provides common feature to be inherited by the 
    loss type objects, such as Square, Normal , etc. 

    Parameters
    ----------
    y: array like
        observations
    state_weight: array like
        weight for the observations
    '''
    def __init__(self, y, weights=None):
        # There may be some some overlapp with these checks on weights and y within
        # the base loss class object in base_loss.py, thus these checks maybe redundent (unless this
        # module is being used without base_loss.py.
        #Checks o n y:
        self._y = check_array_type(y)
        #Checks on weights. 
        if weights is None:
            self._numVar = 0
            self._w = np.ones(self._y.shape)
        else:
            self._w = check_array_type(weights)
        if np.any(weights<0):
            raise ValueError('No elements in numpy array of weights should be negative')
        if len(self._w.shape) > 1:
            if 1 in self._w.shape:
                self._w = self._w.flatten()
        assert self._y.shape == self._w.shape, \
            "Input weight not of the same size as y"
    
    def residual(self, yhat, weighting_applied = True):
        '''
        Raw residuals returned if weighting_applied = False, else
        the weighted residuals.

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals returned.

        Returns
        -------
        :math:`y_{i} - \\hat{y}_{i}`

        '''
        if isinstance(weighting_applied,bool)=False:
            raise TypeError('weighting_applied should be boolean')
            
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                resid = self._y - yhat.ravel()
            else:
                resid = self._y - yhat
        else:
            resid = self._y - yhat
        if weighting_applied = True:
            resid *= self._w
            
        return resid
        
    

class Square(baseloss_type):
    '''
    Square loss object

    Parameters
    ----------
    y: array like
        observations
    '''

    def __init__(self, y, weights=None):
        super().__init__(y, weights=None)
        self.loss(self._y)

    def loss(self, yhat, weighting_applied = True):
        '''
        Loss under square loss.  Not really saying much here

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.
        

        Returns
        -------
        :math:`\\sum_{i=1}^{n} (y-\\hat{y})^{2}`
        '''
        return (self.residual(yhat,weighting_applied)**2).sum()

    def diff_loss(self, yhat,weighting_applied=True):
        '''
        Derivative under square loss.  Assuming that we are solving
        the minimization problem i.e. our objective function is the
        negative of the log-likelihood

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        :math:`-2(y_{i} - \\hat{y}_{i})`
        '''
        return -2*self.residual(yhat,weighting_applied))

    def diff2Loss(self, yhat):
        '''
        Twice derivative of the square loss.  Which is simply 2.

        Parameters
        ----------
        yhat: array like
            observations

        Returns
        -------
        array with values of 2:
            either a scalar, vector or matrix depending on the shape of
            of the input yhat
        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return 2*np.ones(yhat.shape)

class Normal(baseloss_type):
    '''
    Normal distribution loss object

    Parameters
    ----------
    y: array like
        observation
    sigma: float
        standard deviation
    '''       
    def __init__(self, y, sigma=1.0,weight=None):
        super().__init__(y, weights=None)
        err_str = "Standard deviation not of the correct "
        if isinstance(sigma, np.ndarray):
            if np.any(sigma<0):
                raise ValueError('No elements in numpy array of sigma values should be negative')
            elif len(sigma.shape) > 1:
                if 1 in sigma.shape:
                    sigma = sigma.flatten()
                if y.shape == sigma.shape:
                    self._sigma = sigma
                else:
                    raise InitializeError(err_str + "size")
            else:
                if y.shape == sigma.shape:
                    self._sigma = sigma
                else:
                    raise InitializeError(err_str + "size")
        elif sigma is None or sigma == 1.0:
            self._sigma = np.ones(self._y.shape)
        elif isinstance(sigma, (int, float)):
            if sigma <0:
                raise ValueError('Sigma should not be negative')
            else:
                self._sigma = sigma*np.ones(self._y.shape)
        else:
            raise TypeError(err_str + "type")

        self._sigma2 = self._sigma**2
        self.loss(self._y)

    def loss(self, yhat,weighting_applied=True):
        '''
        The loss under a normal distribution.  Defined as the
        negative log-likelihood here.

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        negative log-likelihood, :math:`\\mathcal\\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-\\frac{(y-\\hat{y})^{2}}{2\\sigma^{2}}`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
         
        # Calculate negative likelihood (depending on weighting of residuals).
        logpdf_p1= -np.log(2)
        logpdf_p2= np.log(2)/2
        logpdf_p3= -np.log(pi)/2
        logpdf_p4= np.log(1/self._sigma)
        logpdf_p5_alt=-self.residual(yhat,weighting_applied)**2/(2*sigma**2)
        return (-(logpdf_p1+logpdf_p2+logpdf_p3+logpdf_p4+logpdf_p5_alt)).sum()

    def diff_loss(self, yhat,weighting_applied=True):
        '''
        Derivative of the loss function which is
        :math:`\\sigma^{-1}(y - \\hat{y})`

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        r: array like
            :math:`\\nabla \\mathcal{L}(\\hat{y}, y)`

        '''
        r = self.residual(yhat,weighting_applied)
        return -r/self._sigma2

    def diff2Loss(self, yhat):
        '''
        Twice derivative of the normal loss.

        Parameters
        ----------
        yhat: array like
            observations

        Returns
        -------
        s: array like
            inverse of the variance with shape = yhat.shape
        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return np.ones(yhat.shape)/self._sigma2
    
class Gamma(baseloss_type):
    '''
    Gamma distribution loss object

    Parameters
    ----------
    y: array like
        observation
    Shape: float 
        shape (a in latex equations)
    '''
    
    def __init__(self, y, shape=2.0, weights=None):
        super().__init__(y, weights=None)
        err_str = "Shape is not of the correct "
        if isinstance(shape, np.ndarray):
            if np.any(shape<0):
                raise ValueError('No elements in numpy array of shape values should be negative')
            elif len(shape.shape) > 1:
                if 1 in shape.shape:
                    shape = shape.flatten()
                if y.shape == shape.shape:
                    self._shape = shape
                else:
                    raise InitializeError(err_str + "size")
            else:
                if y.shape == shape.shape:
                    self._shape = shape
                else:
                    raise InitializeError(err_str + "size")
        elif shape is None or shape == 2.0:
            self._shape = 2*np.ones(self._y.shape)
        elif isinstance(shape, (int, float)):
            if shape <0:
                raise ValueError('Shape should not be negative')
            else:
                self._shape = shape*np.ones(self._y.shape)
        else:
            raise TypeError(err_str + "type")
        self.loss(self._y)
        
    def loss(self, yhat):
        '''
        The loss under a gamma distribution.  Defined as the negative 
        log-likelihood of the gamma distirbution in terms of mean and shape.
        See: Bolker, B. M. (2008). Gamma. In Ecological Models in R (pp. 131–133). Princeton University Press.
             File "Loss function Calculations.ipnyb"

        Parameters
        ----------
        yhat: array like
            prediction
            

        Returns
        -------
        negative log-likelihood, :math:`\\mathcal{L}(\\hat{y}; y,a)`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
                
        return -gamma_mu_shape(x=self._y, mu=yhat,shape=self._shape,log=True).sum()
    
    def diff_loss(self, yhat,weighting_applied=True):
        '''
        Derivative of the loss function with respect to yhat which is
        See: 
            File "Loss function Calculations.ipnyb"
            
        Parameters
        ----------
        yhat: array like
            prediction
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.
            
        Returns
        -------
        first_deriv_yhat: array like
            :math:`\\mathcal\\frac{a \\left(\\hat{y} - y\\right)}{\\hat{y}^{2}}`

        '''
        return self._shape*-self.residual(yhat,weighting_applied)/yhat**2

    def diff2Loss(self, yhat):
        '''
        Twice derivative of the loss function with respect to yhat.
        See: 
            Jupiter notebook "Loss function Calculations.ipnyb"
        
        Parameters
        ----------
        yhat: array like
            observation
            

        Returns
        -------
        scnd_deriv_yhat: array like
            :math:`\\mathcal\\frac{a \\left(- \\hat{y} + 2 y\\right)}{\\hat{y}^{3}}`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        y = self._y
        shape = self._shape
        
        return shape*(-yhat+2*y)/yhat**3 

class Poisson(baseloss_type):
    '''
    Poisson distribution loss object

    Parameters
    ----------
    y: array like
        observation
    '''

    def __init__(self, y, weights=None):
        super().__init__(y, weights=None)
        self.loss(self._y)

    def loss(self, yhat):
        '''
        The loss under a normal distribution.  Defined as the
        negative log-likelihood here.

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        negative log-likelihood, :math:`\\mathcal{L}(\\hat{y}, y)`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        # note that we input the standard deviation here
        return (-dpois(self._y, yhat, True)).sum()

    def diff_loss(self, yhat,weighting_applied=True):
        '''
        Derivative of the loss function, :math:`1 - y\\hat{y}^{-1}`

        Parameters
        ----------
        yhat: array like
            observation
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        :math:`\\nabla \\mathcal{L}(\\hat{y},y)`

        '''
        
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
                
        r = self.residual(yhat,weighting_applied)
        return -r/yhat

    def diff2Loss(self, yhat):
        '''
        Twice derivative of the Poisson loss.

        Parameters
        ----------
        yhat: array like
            observations

        Returns
        -------
        s: array like
            :math:`\\frac{y}{\\hat{y}^{2}}` with shape = yhat.shape
        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return self._y/(yhat**2)

class NegBinom(baseloss_type):
    '''
    Negative Binomial distribution loss object

    Parameters
    ----------
    y: array like
        observation
    k: float 
        Overdispersion parameter (k=mean+mean(mean/variance))
    '''
    
    def __init__(self, y,  k=1.0, weights=None):
        super().__init__(y, weights=None)
        err_str = "k (the overdispersion parameter) is not of the correct "
        if isinstance(k, np.ndarray):
            if np.any(k<0):
                raise ValueError('No elements in numpy array of shape values should be negative')
            elif len(k.shape) > 1:
                if 1 in k.shape:
                    k = k.flatten()
                if y.shape == k.shape:
                    self._k = k
                else:
                    raise InitializeError(err_str + "size")
            else:
                if y.shape == k.shape:
                    self._k = k
                else:
                    raise InitializeError(err_str + "size")
        elif k is None or k == 1.0:
            self._k = np.ones(self._y.shape)
        elif isinstance(k, (int, float)):
            if k <0:
                raise ValueError('k should not be negative')
            else:
                self._k = k*np.ones(self._y.shape)
        else:
            raise TypeError(err_str + "type")
            
        self.loss(self._y)
        
    def loss(self, yhat):
        '''
        The loss under a Negative Binomial distribution.  Defined as the
        negative log-likelihood of the Negative Binomial 2 distribution.

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        negative log-likelihood, :math:`\\mathcal{L}(\\hat{y}; y,k)`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
                
        return (-dnbinom(self._y, mu=yhat,size=self._k,log=True)).sum()
    
    def diff_loss(self, yhat,weighting_applied=True):
        '''
        Derivative of the loss function with respect to yhat which is
        See: 
            Jupiter notebook "Loss function Calculations.ipnyb"
            Bolker, B. M. (2008). Negative Binomial. In Ecological Models in R (pp. 124–126). Princeton University Press

        Parameters
        ----------
        yhat: array like
            observation
            
        k: array like
            observation
            
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        first_deriv_yhat: array like
            :math:`\\frac{k(\\hat{y}-y)}{\\hat{y}(k + \\hat{y})}`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
                
        y = self._y
        k = self._k
        r = self.residual(yhat,weighting_applied)
        first_derivs_yhat = k*-r/(yhat*(k+yhat))
        return first_derivs_yhat

    def diff2Loss(self, yhat,weighting_applied=True):
        '''
        Twice derivative of the loss function with respect to yhat.
        See: 
            Jupiter notebook "Loss function Calculations.ipnyb"
            Bolker, B. M. (2008). Negative Binomial. In Ecological Models in R (pp. 124–126). Princeton University Press

        Parameters
        ----------
        yhat: array like
            observation
            
        k: array like
            observation
        
        weighting_applied: boolean
            If True multiplies array of residuals by weightings, else raw 
            residuals are used.

        Returns
        -------
        scnd_deriv_yhat: array like
            :math:`\\frac{k(\\hat{y}(k + \\hat{y}) + \\hat{y}(y -\\hat{y}) + (k + \\hat{y})(y - \\hat{y})}{\\hat{y}^{2}(k + \\hat{y})^{2}}`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        y = self._y
        k = self._k
        r = self.residual(yhat,weighting_applied)
        scnd_derivs_yhat_p1= k
        scnd_derivs_yhat_p2= yhat**(-2)
        scnd_derivs_yhat_p3= (k + yhat)**(-2)
        scnd_derivs_yhat_p4= r*yhat + r*(k + yhat) + yhat*(k + yhat)
        scnd_derivs_yhat= scnd_derivs_yhat_p1*scnd_derivs_yhat_p2*scnd_derivs_yhat_p3*scnd_derivs_yhat_p4
        return scnd_derivs_yhat    