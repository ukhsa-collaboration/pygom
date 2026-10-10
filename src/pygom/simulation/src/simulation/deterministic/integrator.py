"""
Deterministic solver class
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
from dataclasses import dataclass

# ------------------
# Solver diagnostics
# ------------------

# TODO: put dataclass in other file?

@dataclass
class DeterministicOutput:
    t: np.ndarray
    y: np.ndarray
    event_counts: np.ndarray
    scipy_out: Any

class DeterministicSolver:
    def __init__(self, ode, n_state, event_rates=None, n_event=None, jac_ode=None, jac_events=None):
        self.ode = ode
        self.n_state = n_state

        # "Pure ODE" means that:
        # i) the user has not specified the underlying compartmental model, OR
        # ii) the ode system is not the result of a compartmental model (e.g. Newton laws of motion)
        self.pure_ode = True

        if (n_event is not None) and (n_event != 0):
            self.n_event = n_event

            if event_rates is None:
                raise(ValueError("event_rates must be specified alongside n_event"))

            self.event_rates = event_rates
            self.jac_events = jac_events
            self.pure_ode = False

        self.jac_ode = jac_ode     

    def augmented_ode(self, t, y_aug):
        """
        Augment state vector with transition rates
        y_aug = [ [states], [transitions] ]
        """
        y = y_aug[:self.n_state]
        
        # rate of change in states
        dy_dt = self.ode(t, y)
        
        # rate of change of event counts
        de_dt = self.event_rates(t, y)
        
        return np.concatenate([dy_dt, de_dt])
    
    def augmented_jacobian(self, t, y_aug):
        """
        Augment state jacobian matrix with transition rate jacobian matrix
        jac_aug = [ [jac_states], [0]
                    [0],          [jac_transitions] ]
        """

        # TODO: be explicit with sparsity
        # assert jac_x.shape == (self.n_state, self.n_state)
        # assert jac_t.shape[1] == self.n_state
        # assert jac_t.shape[0] == len(x_aug) - self.n_stat

        y = y_aug[:self.n_state]
        
        jac_y = self.jac_ode(t, y)             # states
        jac_e = self.jac_events(t, y)       # event_counts

        jac_aug = np.zeros( (len(y_aug), len(y_aug)) )
        jac_aug[:self.n_state, :self.n_state] = jac_y
        jac_aug[self.n_state:, :self.n_state] = jac_e
        
        return jac_aug

    def integrate(self,
                  t_span,
                  y0,
                  method,
                  t_eval,
                  dense_output,
                  events,
                  vectorized,
                  args,
                  **options):

        # If system is just and ODE, solve it.
        # If system has an underlying compartmental model, augment it with
        # event occurance equations and solve the augented system

        if self.pure_ode:
            if method in ['Radau', 'BDF', 'LSODA']:
                options.setdefault("jac", self.jac_ode)

            sol = solve_ivp(
                fun = self.ode,
                t_span = t_span,
                y0 = y0,
                method=method,
                t_eval=t_eval,
                dense_output=dense_output,
                events=events,
                vectorized=vectorized,
                args=args,
                **options
            )

            t = sol.t
            y = sol.y

            return DeterministicOutput(
                t=t,
                y=y.T,
                event_counts=None,
                scipy_out=sol)

        else:
            e0 = np.zeros(self.n_event)
            y_aug0 = np.concatenate([y0, e0])

            if method in ['Radau', 'BDF', 'LSODA']:
                options.setdefault("jac", self.augmented_jacobian)

            sol = solve_ivp(
                fun = self.augmented_ode,
                t_span = t_span,
                y0 = y_aug0,
                method=method,
                t_eval=t_eval,
                dense_output=dense_output,
                events=events,
                vectorized=vectorized,
                args=args,
                **options
            )

            t = sol.t
            y_aug = sol.y

            y = y_aug[:self.n_state, :]
            event_counts = y_aug[self.n_state:, :]
            event_counts = np.diff(event_counts, axis=1)

            return DeterministicOutput(
                t=t,
                y=y.T,
                event_counts=event_counts.T,
                scipy_out=sol)