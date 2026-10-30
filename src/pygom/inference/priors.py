"""
Prior distributions (currently, only ABC needs these)
"""
import numpy as np
from scipy.stats import uniform, norm

# TODO: priors with limits?

class Prior:
    def sample(self, rng):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.logpdf(x))

class UniformPrior(Prior):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, rng):
        return rng.uniform(self.lower, self.upper)

    def pdf(self, x):
        return uniform(loc=self.lower, scale=self.upper-self.lower).pdf(x)

    def logpdf(self, x):
        return uniform(loc=self.lower, scale=self.upper-self.lower).logpdf(x)

class NormalPrior(Prior):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, rng):
        return rng.normal(self.mean, self.std)

    def pdf(self, x):
        return norm(loc=self.mean, scale=self.std).pdf(x)

    def logpdf(self, x):
        return norm(loc=self.mean, scale=self.std).logpdf(x)


# TODO: Include other priors, maybe no need to write out ourselves like above