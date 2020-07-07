"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    The different loss types.  Such as thought based on
    parametric distributions.

"""

__all__ = [
    'Square',
    'Normal',
    'Poisson'
    ]

import numpy as np

from pygom.model._model_errors import InitializeError
from pygom.model.ode_utils import check_array_type
from pygom.utilR.distn import dnorm, dpois

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
        The loss under a Poisson distribution.  Defined as the
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

