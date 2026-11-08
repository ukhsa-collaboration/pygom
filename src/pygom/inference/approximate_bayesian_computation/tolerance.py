"""
Tolerance
"""

import numpy as np

class Tolerance:

    def tolerance(weights):
        return NotImplementedError


class Quantile(Tolerance):

    def __init__(self, quantile):
        self.quantile = quantile

    def tolerance(self, weights):
        return np.quantile(weights, q=self.quantile)

class Geometric(Tolerance):

    def __init__(self, quantile):
        self.quantile = quantile

    def tolerance(self, weights):
        return np.quantile(weights, q=self.quantile)