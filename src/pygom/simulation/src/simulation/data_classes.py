"""
Data classes for stochastic simulation outputs
"""

from dataclasses import dataclass
import numpy as np

# Stochastic simulation metadata
@dataclass
class SolverDiagnostics:
    zero_rate_termination: bool = False

@dataclass
class SolverConfig:
    pass

@dataclass
class PerformanceMetrics:
    wall_time_seconds: float
    cpu_time_seconds: float | None = None

# Stochastic simulation results
@dataclass
class SimulationResult:
    pass

@dataclass
class TimeSeriesResult(SimulationResult):
    t: np.ndarray               # time points
    y: np.ndarray               # state at times, t
    event_counts: np.ndarray    # number of times each event occured between successive timepoints

@dataclass
class EventSeriesResult(SimulationResult):
    t: np.ndarray           # time points
    y: np.ndarray           # state at times, t
    event_id: np.ndarray    # id of event which occured at time t

# Combine metadata and results to produce output
@dataclass
class Output:
    result: SimulationResult
    config: SolverConfig
    diag: SolverDiagnostics | None = None
    performance: PerformanceMetrics | None = None
