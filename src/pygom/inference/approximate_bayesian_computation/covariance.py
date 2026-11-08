"""
Calculate covariances
"""

from .particle_distance import Distance
import numpy as np
from .data_classes import Population

class CovarianceMethod:
    """
    Calculate covariance matrix (or matrices)
    """

    def prepare(self, population:Population, tolerance:float):
        """
        Cache objects we only want to calculate once.
        For now, this is just a was to pre-compute sigma
        """
        pass

    def configure(self, unknowns):
        """
        ........
        """
        pass

    def get_sigma(self, ancestor_idx:int):
        """
        Get sigma, if no ancestor_idx specified, assumed to be global
        """
        raise NotImplementedError

class GlobalCovariance(CovarianceMethod):
    """
    Get covariance matrix of particles
    """

    def prepare(self, population:Population, tolerance:float):
        self._sigma = self._compute_global_covariances(population, tolerance)

    def _compute_global_covariances(self, population:Population, tolerance:float):
        return np.cov(population.particles.T)

    def get_sigma(self, ancestor_idx:int):
        return self._sigma

class LocalCovariance(CovarianceMethod):
    """
    Get local covariance.
    What constitutes "local" may vary
    """

    def prepare(self, population:Population, tolerance:float):
        self._sigma_list = self._compute_local_covariances(population, tolerance)

    def _compute_local_covariances(self, population:Population, tolerance:float):
        """
        Find the covariance matrix of the M nearest particles to a specified particle
        """
        # Tolerated subset of particles
        accepted_idx = population.distances < tolerance
        particles_tol = population.particles[accepted_idx]
        weights_tol = population.weights[accepted_idx]
        weights_tol = weights_tol / weights_tol.sum()

        sigma_list = []
        for centre in population.particles:
            diff = particles_tol - centre
            sigma = np.einsum("ij,ik,i->jk", diff, diff, weights_tol)
            sigma += np.eye(sigma.shape[0]) * 1e-8
            sigma_list.append(sigma)

        return sigma_list

    def get_sigma(self, ancestor_idx:int):
        return self._sigma_list[ancestor_idx]

class KNNCovariance(CovarianceMethod):
    """
    Get k nearest neighbours covariance
    """

    def __init__(
            self,
            k: int|None=None,
            distance_method: Distance|None=None
        ):

        if distance_method is None:
            raise ValueError("Must set distance method")
        if k is None:
            raise ValueError("Must set value for k nearest neigbours")
        if k < 2:
            raise ValueError(f"k={k} must be >=2")

        self.distance_method = distance_method
        self.k = k

    def configure(self, unknowns):
        self.distance_method.configure(unknowns)

    def prepare(self, population:Population, tolerance:float):
        n_particles = len(population.particles)
        if self.k >= n_particles:
            raise ValueError(f"k={self.k} must be less than the number of particles ({n_particles})")

        self.distance_method.prepare(population.particles, population.weights)
        self._sigma_list = self._compute_local_covariances(population, tolerance)

    def _compute_local_covariances(self, population:Population, tolerance:float):
        """
        Find the covariance matrix of the M nearest particles to a specified particle
        """

        sigma_list = []
        for idx in range(len(population.particles)):
            nearest_neighbour_idx = self.distance_method.nearest_neighbours_idx(population.particles, idx, self.k)
            sigma = np.cov(population.particles[nearest_neighbour_idx].T)
            sigma += np.eye(sigma.shape[0]) * 1e-8
            sigma_list.append(sigma)

        return sigma_list

    def get_sigma(self, ancestor_idx:int):
        return self._sigma_list[ancestor_idx]