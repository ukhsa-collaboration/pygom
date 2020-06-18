"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    The loss functions that "implements" the class BaseLoss, if Python
    has such a thing.  Overrides the method _setLossType

"""

__all__ = [
    'SquareLoss',
    'NormalLoss',
    'PoissonLoss',
    'GammaLoss',
    'NegBinomLoss'
    ]

from pygom.loss.base_loss import BaseLoss
from pygom.loss.loss_type import Normal, Square, Poisson, Gamma, NegBinom
from pygom.model.ode_utils import check_array_type

class SquareLoss(BaseLoss):
    '''
    The square loss function
    '''
    def __init__(self, theta, ode, x0, t0, t, y, state_name,
                 state_weight=None, target_param=None, target_state=None):
        super(SquareLoss, self).__init__(theta, ode, x0, t0, t, y,
                                         state_name, state_weight,
                                         target_param, target_state)

    def __repr__(self):
        return "SquareLoss" + self._get_model_str()

    def _setLossType(self):
        self._lossObj = Square(self._y, self._stateWeight)
        return self._lossObj

class NormalLoss(BaseLoss):
    '''
    Realizations from a Normal distribution
    '''
    def __init__(self, theta, ode, x0, t0, t, y, state_name,
                 sigma=None, target_param=None, target_state=None):
        if sigma is None:
            super(NormalLoss, self).__init__(theta, ode, x0, t0, t, y,
                                             state_name, sigma,
                                             target_param, target_state)
        else:
            sigma = check_array_type(sigma)
            super(NormalLoss, self).__init__(theta, ode, x0, t0, t, y,
                                             state_name, 1.0/sigma,
                                             target_param, target_state)

    def __repr__(self):
        return "NormalLoss" + self._get_model_str()

    def _setLossType(self):
        if self._stateWeight is None:
            return Normal(self._y, 1.0)
        else:
            if len(self._stateWeight.shape) > 1:
                if 1 in self._stateWeight.shape:
                    return Normal(self._y, 1.0/self._stateWeight.flatten())
                else:
                    return Normal(self._y, 1.0/self._stateWeight)
            else:
                return Normal(self._y, 1.0/self._stateWeight)
            
class GammaLoss(BaseLoss):
    '''
    Realizations from a Gamma distribution taking parameters mean and shape.
    '''
    def __init__(self, theta, ode, x0, t0, t, y, state_name,state_weight=None,
                 shape=2, target_param=None, target_state=None):
        self._shape = shape
        super(GammaLoss, self).__init__(theta, ode, x0, t0, t, y,
                                             state_name, state_weight,
                                             target_param, target_state)
            
    def __repr__(self):
        return "GammaLoss" + self._get_model_str()

    def _setLossType(self):
        return Gamma(self._y,self._shape,self._state_weight))

class PoissonLoss(BaseLoss):
    '''
    Realizations from a Poisson distribution
    '''
    def __init__(self, theta, ode, x0, t0, t, y, state_name,state_weight=None,
                 target_param=None, target_state=None):
        super(PoissonLoss, self).__init__(theta, ode, x0, t0, t, y, state_name,
                                          None, target_param, target_state)
    def __repr__(self):
        return "PoissonLoss" + self._get_model_str()

    def _setLossType(self):
        return Poisson(self._y,self._state_weight))
    
class NegBinomLoss(BaseLoss):
    '''
    Realizations from a Negative Binomial distribution
    '''
    def __init__(self, theta, ode, x0, t0, t, y, state_name,state_weight=None,
                 k=1, target_param=None, target_state=None):
        self._k = k
        super(NegBinomLoss, self).__init__(theta, ode, x0, t0, t, y,
                                             state_name, state_weight,
                                             target_param, target_state)
            
    def __repr__(self):
        return "NegBinomLoss" + self._get_model_str()

    def _setLossType(self):
        return NegBinom(self._y,self._k,self._state_weight)
