"""
Data classes which are used configure stochastic solvers
"""

from dataclasses import dataclass, field
from typing import Union, Callable

from ..data_classes import SolverConfig

# =====
# Exact
# =====
# --------------
# method configs
# --------------
class ExactMethodConfig:
    pass

@dataclass
class DirectMethodConfig(ExactMethodConfig):
    pass

@dataclass
class FirstReactionMethodConfig(ExactMethodConfig):
    pass

# ===
# Tau 
# ===
# ----------------
# method configs
# ----------------
class TauMethodConfig:
    pass

@dataclass
class FixedTauConfig(TauMethodConfig):
    tau: float

@dataclass
class Cao2006TauConfig(TauMethodConfig):
    epsilon: float
    transition_mean_func: Callable
    transition_var_func: Callable

@dataclass
class UKHSA2026TauConfig(TauMethodConfig):
    epsilon: float
    timestep_mean_func: Callable
    timestep_var_func: Callable

@dataclass
class Alternative2026TauConfig(TauMethodConfig):
    epsilon: float

# ---------------
# refiner configs
# ---------------
class TauRefinerConfig:
    pass

@dataclass
class ProbabilisticRefinerConfig(TauRefinerConfig):
    max_retries: int = 10
    acceptable_prob_misstep: float = 0.05
    factor_min: float = 0.1
    factor_max: float = 0.9

@dataclass
class NoRefinerConfig(TauRefinerConfig):
    pass

# ===============
# checker configs
# ===============
class CheckerConfig:
    pass

@dataclass
class CriticalReactionConfig(CheckerConfig):
    pass

@dataclass
class ForbiddenStateConfig(CheckerConfig):
    pass

@dataclass
class NoCheckConfig(CheckerConfig):
    pass


# ==============
# Solver configs
# ==============
@dataclass
class ExactConfig(SolverConfig):
    method: Union["DirectMethodConfig", "FirstReactionMethodConfig"]

@dataclass
class TauConfig(SolverConfig):
    method: "TauMethodConfig"

    checker: "CheckerConfig" = field(
        default_factory=CriticalReactionConfig
    )
    refiner: "TauRefinerConfig" = field(
        default_factory=NoRefinerConfig
    )

    retry_max: int = 10
    tau_rescale: float = 0.5