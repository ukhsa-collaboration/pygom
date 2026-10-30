import numpy as np
from scipy.stats import multivariate_normal
import time
from dataclasses import dataclass
from joblib import Parallel, delayed
import os
from .factory import build_covariance
from .data_classes import Population

# TODO:
# 1) Logging instead of printing progress
# 2) Parallel ABC (current joblib implementaion is a bit of an experiment).
#    Probably worth building general parallel backend to be used across pygom
# 3) Sample parameters in transformed space (e.g. log transform)
# 4) Sampling kernel: now only "gaussian". Could add other distributions e.g. "student_t"?
# 6) Tolerance method: now only "quantile". Could add pre-set "fixed" tolerances, "ESS based" tolerances?

## Dataclasses to store particle genration information ##

@dataclass
class Generation:
    population: Population                  # population of particles
    generation_number: int                  # ABC generation, starting at 0th
    out_of_bounds_rejections: int           # Number of times a proposed particle is rejected due to being in invalid region of parameter space
    above_tolerance_rejections: int         # Number of times a proposed particle is rejected due to distance being above tolerance threshold
    attempts: int                           # Total number of attempts taken to generate ful set of accepted particles
    tolerance: float                        # Tolerance for this generation
    ess: float                              # Effective Sample Size

## Algorithm ##

class ABC:
    def __init__(
        self,

        # sampling options
        n_particles,
        quantile=0.5,
        initial_tolerance=np.inf,
        max_attempts=1_000_000,

        # covariance options
        covariance_method="global",     # global, local, knn

        # nearest neighbour options
        distance_method=None,           # euclidean, scaled, mahalanobis
        k=None,
        lengthscales=None,
        regularisation=None,

        rng=None,
        progress=True,

        # parallelisation
        n_workers=None,
        batch_size=32                   # TODO: Maybe scope to optimise batch size
    ):
        self.n_particles = n_particles
        self.quantile = quantile
        self.initial_tolerance = initial_tolerance
        self.max_attempts = max_attempts

        self.covariance_method = build_covariance(
            covariance_method=covariance_method,
            k=k,
            distance_method=distance_method,
            lengthscales=lengthscales,
            regularisation=regularisation
        )

        self.rng = rng or np.random.default_rng()
        self.progress = progress

        if n_workers is None:
            n_workers = os.cpu_count()
        self.n_workers = n_workers
        self.batch_size = batch_size

    def _initial_generation(self, objective, unknowns):
        # TODO: this is very similar to _next_generation, should we make
        #       this one function and add a flag for gen 0?

        tolerance = self.initial_tolerance

        particles = []
        distances = []

        attempts = 0
        out_of_bounds_rejections = 0
        above_tolerance_rejections = 0

        while len(particles) < self.n_particles:

            # prepare batch of proposals
            proposals = []
            while (len(proposals) < self.batch_size) and (attempts < self.max_attempts):

                attempts += 1

                # propose particle from prior
                theta = np.array([
                    unknown.prior.sample(self.rng)
                    for unknown in unknowns
                ])

                # Check proposal is within allowed range
                if not self._check_within_bounds(unknowns, theta):
                    out_of_bounds_rejections += 1
                    continue

                proposals.append(theta)

            if attempts > self.max_attempts:
                raise RuntimeError(
                    f"Failed to generate enough accepted particles after {self.max_attempts} attempts"
                )

            # Calculate distances in parallel
            distances_batch = Parallel(n_jobs=self.n_workers, backend="loky")(
                delayed(objective)(theta)
                for theta in proposals
            )

            # Process the distances batch
            for (theta, distance) in zip(proposals, distances_batch):

                if distance > tolerance:
                    above_tolerance_rejections += 1
                    continue

                particles.append(theta)
                distances.append(distance)

                if len(particles) >= self.n_particles:
                    break

        particles = np.asarray(particles)
        distances = np.asarray(distances)
        weights = np.ones(self.n_particles) / self.n_particles      # equal weights
        ess = 1 / sum(weights**2)

        population = Population(
            particles=particles,
            distances=distances,
            weights=weights
        )

        return Generation(
            population=population,
            generation_number=0,
            out_of_bounds_rejections=out_of_bounds_rejections,
            above_tolerance_rejections=above_tolerance_rejections,
            attempts=attempts,
            tolerance=tolerance,
            ess=ess
        )

    def _compute_tolerance(self, distances):
        """
        Get tolerance via quantile of distance distribution
        """
        return np.quantile(distances, self.quantile)

    def _compute_weight(self, theta, population:Population, unknowns, sigma):

        prior_density = np.prod([
            unknown.prior.pdf(value)
            for unknown, value in zip(unknowns, theta)
        ])

        kernel_density = 0.0

        for particle, weight in zip(population.particles, population.weights):
            kernel_density += weight * multivariate_normal.pdf(theta, mean=particle, cov=sigma)

        return prior_density / kernel_density

    def _propose_particle(self, population:Population):

        idx = self.rng.choice(self.n_particles, p=population.weights)

        centre = population.particles[idx]
        sigma = self.covariance_method.get_sigma(idx)

        theta = self.rng.multivariate_normal(centre, sigma)

        return theta, idx

    def _check_within_bounds(self, unknowns, theta):
        """
        Check that particle coordinates does not fall outside of allowed range
        imposed either by limits or prior distribution
        """
        for u, value in zip(unknowns, theta):
            if (
                (value <= u.lower) or
                (value >= u.upper) or
                (u.prior.pdf(value) == 0)
                ):
                return False
        
        return True

    def _next_generation(self, current_generation:Generation, objective, unknowns):
        """
        Move the particles forward one step from current_generation to the next
        """

        current_population = current_generation.population
        next_genertation_number = current_generation.generation_number + 1
        tolerance = self._compute_tolerance(current_population.distances)

        # Precompute sigma values
        self.covariance_method.prepare(current_population, tolerance)

        # output containers
        new_particles = []
        new_weights = []
        new_distances = []

        # diagnostic counters
        attempts = 0
        out_of_bounds_rejections = 0
        above_tolerance_rejections = 0

        while len(new_particles) < self.n_particles:
            # prepare batch of proposals
            proposals = []
            ancestor_indices = []
            while (len(proposals) < self.batch_size) and (attempts < self.max_attempts):

                attempts += 1

                theta, ancestor_idx = self._propose_particle(current_population)

                if not self._check_within_bounds(unknowns, theta):
                    out_of_bounds_rejections += 1
                    continue

                proposals.append(theta)
                ancestor_indices.append(ancestor_idx)

            if attempts > self.max_attempts:
                raise RuntimeError(
                    f"Failed to generate enough accepted particles after {self.max_attempts} attempts"
                )

            # Calculate distances in parallel
            # distances_batch = list(executor.map(objective, proposals))

            distances_batch = Parallel(n_jobs=self.n_workers, backend="loky")(
                delayed(objective)(theta)
                for theta in proposals
            )

            # Process the distances batch
            for (theta, ancestor_idx, distance) in zip(proposals, ancestor_indices, distances_batch):

                if distance > tolerance:
                    above_tolerance_rejections += 1
                    continue

                sigma = self.covariance_method.get_sigma(ancestor_idx)

                weight = self._compute_weight(theta, current_population, unknowns, sigma)

                new_particles.append(theta)
                new_weights.append(weight)
                new_distances.append(distance)

                if len(new_particles) >= self.n_particles:
                    break

        new_distances = np.asarray(new_distances)
        new_weights = np.asarray(new_weights)
        new_weights /= new_weights.sum()

        # ESS
        ess = 1 / sum(new_weights**2)

        population = Population(
            particles=np.asarray(new_particles),
            distances=new_distances,
            weights=new_weights
        )

        return Generation(
            population=population,
            generation_number=next_genertation_number,
            out_of_bounds_rejections=out_of_bounds_rejections,
            above_tolerance_rejections=above_tolerance_rejections,
            attempts=attempts,
            tolerance=tolerance,
            ess=ess
        )

    def _progress_statement(self, generation:Generation, time):
        """
        Update to be printed after successful generation
        """

        # -----------------------------------------------
        # Break down reasons for particles being rejected
        # -----------------------------------------------

        out_of_bounds_rejections = generation.out_of_bounds_rejections
        above_tolerance_rejections = generation.above_tolerance_rejections
        attempts = generation.attempts

        # Total
        total_rejections = out_of_bounds_rejections + above_tolerance_rejections
        acceptance_rate_total = (attempts - total_rejections) / attempts

        # Out of bounds
        rejection_rate_limits = out_of_bounds_rejections / attempts

        # Above tolerance
        tolerance_attempts = attempts - out_of_bounds_rejections        # (subset to those which got past the out of bounds checks)
        rejection_rate_tolerance = above_tolerance_rejections / tolerance_attempts

        return (
            f"Generation: {generation.generation_number}\n"
            f"Tolerance: {generation.tolerance}\n"
            f"Accepted: {(attempts - total_rejections)} / {attempts} ({100*acceptance_rate_total:.1f}%)\n"
            f"Rejected (outside allowed limits): {out_of_bounds_rejections} / {attempts} ({100*rejection_rate_limits:.1f}%)\n"
            f"Rejected (above distance tolerance): {above_tolerance_rejections} / {tolerance_attempts} ({100*rejection_rate_tolerance:.1f}%)\n"
            f"ESS: {generation.ess}\n"
            f"Time taken: {time} seconds\n\n"
        )

    def refine_posterior(self, objective, unknowns, n_generations, previous_generations=None):
        """
        Refine posterior distribution by progressing through n_generations
        """
        self.covariance_method.configure(unknowns)

        # Resume from previous run ....
        if previous_generations is not None:
            print(f"Continuing ABC from generation {previous_generations[-1].generation_number}")
            n_generations += 1
            generation = previous_generations[-1]
            out = previous_generations
        # .... or start from scratch
        else:
            t1 = time.time()
            generation = self._initial_generation(objective, unknowns)
            t2 = time.time()
            out = [generation]
        
            if self.progress:
                print(self._progress_statement(generation, t2-t1))

        for gen in range(n_generations - 1):
            t1 = time.time()
            generation = self._next_generation(generation, objective, unknowns)
            t2 = time.time()
            out.append(generation)

            if self.progress:
                print(self._progress_statement(generation, t2-t1))

        return out