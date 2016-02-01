"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    The loss functions that "implements" the class BaseLoss, if Python
    has such a thing.  Overrides the method _setLossType

"""

__all__ = [
    'SquareLoss',
    'NormalLoss',
    'PoissonLoss'
    ]

from pygom.loss.base_loss import BaseLoss
from pygom.loss.loss_type import Normal, Square, Poisson
from pygom.model.ode_utils import checkArrayType

class SquareLoss(BaseLoss):
    '''
    The square loss function
    '''
    def __init__(self, theta, ode, x0, t0, t, y, stateName,
                 stateWeight=None, targetParam=None, targetState=None):
        super(SquareLoss, self).__init__(theta, ode, x0, t0, t, y, stateName,
                                         stateWeight, targetParam, targetState)

    def _setLossType(self):
        self._lossObj = Square(self._y, self._stateWeight)
        return self._lossObj

class NormalLoss(BaseLoss):
    '''
    Realizations from a Normal distribution
    '''
    def __init__(self, theta, ode, x0, t0, t, y, stateName,
                 sigma=None, targetParam=None, targetState=None):
        if sigma is None:
            super(NormalLoss, self).__init__(theta, ode, x0, t0, t, y, stateName,
                                             sigma, targetParam, targetState)
        else:
            sigma = checkArrayType(sigma)
            super(NormalLoss, self).__init__(theta, ode, x0, t0, t, y, stateName,
                                             1/sigma, targetParam, targetState)

    def _setLossType(self):
        if self._stateWeight is None:
            return Normal(self._y, 1.0)
        else:
            if len(self._stateWeight.shape) > 1:
                if 1 in self._stateWeight.shape:
                    return Normal(self._y, 1/self._stateWeight.flatten())
                else:
                    return Normal(self._y, 1/self._stateWeight)
                    # raise RuntimeError("Normal loss can only use standard deviation")
            else:
                return Normal(self._y, 1/self._stateWeight)

class PoissonLoss(BaseLoss):
    '''
    Realizations from a Poisson distribution
    '''
    def __init__(self, theta, ode, x0, t0, t, y, stateName,
                 targetParam=None, targetState=None):
        super(PoissonLoss, self).__init__(theta, ode, x0, t0, t, y, stateName,
                                          None, targetParam, targetState)

    def _setLossType(self):
        return Poisson(self._y)

