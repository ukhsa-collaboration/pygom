"""
Classes to calculate distances between particles

Useful to obtain nearest neigbours
"""

import numpy as np

class Distance:
    """
    Ways to calculate geometric distances between particles
    NOTE: this is not the same as the distance of each particle
          which relates to the objective function
    """

    def configure(self, unknowns):
        pass

    def prepare(self, particles, weights=None):
        """
        Calculate and cache any objects which will be repeatedly used,
        but only need calculation once.
        """
        pass

    def pairwise_distance(self, x, y):
        """
        Distance between two particles
        
        x: numpy.ndarray
        y: numpy.ndarray
        """
        raise NotImplementedError

    def distances_from(self, particles, idx):
        """
        Distances from particle with index idx to every particle,
        including the particle itself.
        """
        raise NotImplementedError

    def nearest_neighbours_idx(self, particles, idx, k):
        """
        Return indices of k nearest neighbours,
        excluding the particle itself.
        """

        distances = self.distances_from(particles, idx)
        neighbours = np.argsort(distances)
        neighbours = neighbours[neighbours != idx]

        return neighbours[:k]

class EuclidDistance(Distance):
    """
    Euclidean distance:

    d(x, y) = sqrt( Σ_i (x_i - y_i)^2 )
    """
    def pairwise_distance(self, x, y):
        diff = x - y
        return np.sqrt(diff @ diff)

    def distances_from(self, particles, idx):
        return np.linalg.norm(particles - particles[idx], axis=1)

class ScaledDistance(Distance):
    """
    Each dimension, i, is scaled by a characteristic lengthscale, L_i.
    Scaled distance is then Euclidean distance in these normalised units

    d(x, y) = sqrt( Σ_i ((x_i - y_i)/L_i)^2 )
    """

    def __init__(self, lengthscales):
        self.lengthscales = lengthscales

    def configure(self, unknowns):

        if self.lengthscales is not None:
            return

        lengthscales = []
        for unknown in unknowns:
            if np.isfinite(unknown.lower) and np.isfinite(unknown.upper):
                lengthscales.append(unknown.upper - unknown.lower)
            else:
                raise ValueError(
                    f"Cannot automatically determine lengthscale for unknown '{unknown.id}'. "
                     "Specify lengthscales explicitly, or specify upper and lower bounds."
                )

        self.lengthscales = np.asarray(lengthscales)

    def prepare(self, particles, weights=None):
        self._particles_scaled = particles / self.lengthscales

    def pairwise_distance(self, x, y):
        x_scaled = x / self.lengthscales
        y_scaled = y / self.lengthscales

        diff = x_scaled - y_scaled
        return np.sqrt(diff @ diff)

    def distances_from(self, particles, idx):
        return np.linalg.norm(self._particles_scaled - self._particles_scaled[idx], axis=1)
    
class MahalanobisDistance(Distance):
    """
    Mahalanobis distance using the weighted covariance
    of the current particle population.

    d(x, y) = sqrt( (x-y)^T Σ^{-1} (x-y) )
    """

    def __init__(self, regularisation=1e-10):
        self.regularisation = regularisation
        self._inv_cov = None

    def prepare(self, particles, weights=None):
        """
        Estimate covariance and cache its inverse.
        """

        if weights is None:
            cov = np.cov(particles.T)
        else:
            weights = weights / weights.sum()   # just in case not normalised
            mean = np.average(particles, axis=0, weights=weights)
            centred = particles - mean
            cov = np.einsum("i,ij,ik->jk", weights, centred, centred)

        # numerical stabilisation
        cov += self.regularisation * np.eye(cov.shape[0])
        self._inv_cov = np.linalg.inv(cov)

    def pairwise_distance(self, x, y):
        if self._inv_cov is None:
            raise(RuntimeError("prepare() must have been called previously"))

        diff = x - y

        return np.sqrt(diff @ self._inv_cov @ diff)

    def distances_from(self, particles, idx):
        if self._inv_cov is None:
            raise(RuntimeError("prepare() must have been called previously"))
        diff = particles - particles[idx]
        tmp = diff @ self._inv_cov

        return np.sqrt(np.sum(tmp * diff, axis=1))