"""
Data classes shared across ABC files
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Population:
    particles: np.ndarray                   # Parameter values for each particle, shape = (n_particle, n_param)
    distances: np.ndarray                   # Objective function of each particle, shape = (n_particle)
    weights: np.ndarray                     # Weight of each particle, shape = (n_particle)