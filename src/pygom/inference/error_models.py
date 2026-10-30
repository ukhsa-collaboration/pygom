"""
Error models
"""
import numpy as np

class ErrorModel:
    """
    Observed vs predicted error function
    """
    def evaluate(self, observed: float|np.ndarray, predicted: float|np.ndarray) -> float:
        """
        Overloaded by child classes
        """
        raise NotImplementedError

class RelativeSquaredError(ErrorModel):

    def evaluate(self, observed, predicted):

        observed = np.asarray(observed)
        predicted = np.asarray(predicted)

        rel_err = (predicted - observed) / np.maximum(np.abs(observed), 1e-8)

        return np.sum(rel_err**2)

# TODO: plus all others: Square, Normal, NegBinom etc