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

class Square(object):
    '''
    Square loss object

    Parameters
    ----------
    y: array like
        observations
    '''

    def __init__(self, y, weights=None):
        self._y = check_array_type(y)
        self._numObv = len(self._y)

        if weights is None:
            self._numVar = 0
            self._w = np.ones(self._y.shape)
        else:
            self._w = check_array_type(weights)

        if len(self._w.shape) > 1:
            if self._w.shape[1] == 1:
                self._w = self._w.flatten()

        assert self._y.shape == self._w.shape, \
            "Input weight not of the same size as y"

        self.loss(self._y)

    def loss(self, yhat):
        '''
        Loss under square loss.  Not really saying much here

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        :math:`\\sum_{i=1}^{n} (\\hat{y} - y)^{2}`
        '''
        return (self.residual(yhat)**2).sum()

    def diff_loss(self, yhat):
        '''
        Derivative under square loss.  Assuming that we are solving
        the minimization problem i.e. our objective function is the
        negative of the log-likelihood

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        :math:`-2(y_{i} - \\hat{y}_{i})`
        '''
        return -2*self.residual(yhat)

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
        return self._weightedResidual(2*np.ones(yhat.shape))

    def residual(self, yhat):
        '''
        Raw residuals if no weights was initialized, else
        the weighted residuals

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        :math:`y_{i} - \\hat{y}_{i}`

        '''
        return self._weightedResidual(yhat)

    def _weightedResidual(self, yhat):
        '''
        Find the weighted residuals.
        '''
        # resid = self._y - yhat
        # print "In weighted resid"
        # print self._y.shape
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                resid = self._y - yhat.ravel()
            else:
                resid = self._y - yhat
        else:
            resid = self._y - yhat

        return resid * self._w

class Normal(object):
    '''
    Normal distribution loss object

    Parameters
    ----------
    y: array like
        observation
    sigma: float
        standard deviation
    '''

    def __init__(self, y, sigma=1.0):
        err_str = "Standard deviation not of the correct "
        self._y = check_array_type(y)
        if isinstance(sigma, np.ndarray):
            if len(sigma.shape) > 1:
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
        else:
            raise InitializeError(err_str + "type")

        self._sigma2 = self._sigma**2
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
        negative log-likelihood, :math:`\\mathcal{L}(\\hat{y},y)`

        '''
        # note that we input the standard deviation here
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return (-dnorm(self._y, yhat, self._sigma, True)).sum()

    def diff_loss(self, yhat):
        '''
        Derivative of the loss function which is
        :math:`\\sigma^{-1}(y - \\hat{y})`

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        r: array like
            :math:`\\nabla \\mathcal{L}(\\hat{y}, y)`

        '''
        r = self.residual(yhat)
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
        return np.ones(yhat.shape)/self._sigma2

    def residual(self, yhat):
        '''
        Residuals under a normal loss

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        r: array like
            residuals

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                resid = self._y - yhat.ravel()
            else:
                resid = self._y - yhat
        else:
            resid = self._y - yhat

        return resid
    
class Gamma(object):
    '''
    Gamma distribution loss object

    Parameters
    ----------
    y: array like
        observation
    Shape: float 
        shape (a in latex equations)
    '''
    
    def __init__(self, y, shape=2.0):
        err_str = "Shape is not of the correct "
        self._y = check_array_type(y)
        if isinstance(shape, np.ndarray):
            if len(shape.shape) > 1:
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
        elif isinstance(shape, (int, float)) and shape !=2:
            self._shape = shape*np.ones(self._y.shape)
        else:
            raise InitializeError(err_str + "type")

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
    
    def diff_loss(self, yhat):
        '''
        Derivative of the loss function with respect to yhat which is
        See: 
            File "Loss function Calculations.ipnyb"
            
        Parameters
        ----------
        yhat: array like
            prediction
            

        Returns
        -------
        first_deriv_yhat: array like
            :math:`\\mathcal\\frac{a \\left(\\hat{y} - y\\right)}{\\hat{y}^{2}}`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()

        return self._shape*(yhat-self._y)/yhat**2

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
        y = self._y
        shape = self._shape
        
        return shape*(-yhat+2*y)/yhat**3 
    
    def residual(self, yhat):
        '''
        Raw residuals

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        r: array like
            residuals

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return self._y - yhat

class Poisson(object):
    '''
    Poisson distribution loss object

    Parameters
    ----------
    y: array like
        observation
    '''

    def __init__(self, y):
        self._y = check_array_type(y)
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

    def diff_loss(self, yhat):
        '''
        Derivative of the loss function, :math:`1 - y\\hat{y}^{-1}`

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        :math:`\\nabla \\mathcal{L}(\\hat{y},y)`

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return 1 - self._y/yhat

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
        return self.y/(yhat**2)

    def residual(self, yhat):
        '''
        Raw residuals

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        r: array like
            residuals

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return self._y - yhat

class NegBinom(object):
    '''
    Negative Binomial distribution loss object

    Parameters
    ----------
    y: array like
        observation
    k: float 
        Overdispersion parameter (k=mean+mean(mean/variance))
    '''
    
    def __init__(self, y, k=1.0):
        err_str = "k (the overdispersion parameter) is not of the correct "
        self._y = check_array_type(y)
        if isinstance(k, np.ndarray):
            if len(k.shape) > 1:
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
        elif isinstance(k, (int, float)) and k >1:
            self._k = k*np.ones(self._y.shape)
        else:
            raise InitializeError(err_str + "type")

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
    
    def diff_loss(self, yhat):
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
        first_derivs_yhat = (k*(yhat-y))/(yhat*(k+yhat))
        return first_derivs_yhat

    def diff2Loss(self, yhat):
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

        Returns
        -------
        scnd_deriv_yhat: array like
            :math:`\\frac{k(\\hat{y}(k + \\hat{y}) + \\hat{y}(y -\\hat{y}) + (k + \\hat{y})(y - \\hat{y})}{\\hat{y}^{2}(k + \\hat{y})^{2}}`

        '''
        y = self._y
        k = self._k
        scnd_derivs_yhat_p1= k
        scnd_derivs_yhat_p2= yhat**(-2)
        scnd_derivs_yhat_p3= (k + yhat)**(-2)
        scnd_derivs_yhat_p4= yhat*(k + yhat) - yhat*(yhat - y) - (k + yhat)*(yhat - y)
        scnd_derivs_yhat= scnd_derivs_yhat_p1*scnd_derivs_yhat_p2*scnd_derivs_yhat_p3*scnd_derivs_yhat_p4
        return scnd_derivs_yhat    
    
    def residual(self, yhat):
        '''
        Raw residuals

        Parameters
        ----------
        yhat: array like
            observation

        Returns
        -------
        r: array like
            residuals

        '''
        if len(yhat.shape) > 1:
            if 1 in yhat.shape:
                yhat = yhat.ravel()
        return self._y - yhat
