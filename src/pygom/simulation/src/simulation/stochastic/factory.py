"""
Factory to assemble solver components from config classes
"""

from .config import SolverConfig
from .config import TauConfig, FixedTauConfig, Cao2006TauConfig, UKHSA2026TauConfig
from .config import ExactConfig, DirectMethodConfig, FirstReactionMethodConfig
from .core.exact import DirectReaction, FirstReaction
from .core.tau import TauLeap
from .core.tau.method import Fixed, Cao2006, UKHSA2026

from .config import CriticalReactionConfig, NoCheckConfig, ForbiddenStateConfig
from .core.step_checker import CriticalReactionCheck, NoCheck, ForbiddenStateCheck

from .config import NoRefinerConfig, ProbabilisticRefinerConfig
from .core.tau.precaution import ProbabilisticTauPrecaution, NoTauPrecaution

def make_stepper(
    *,
    config: SolverConfig,
    event_rates,
    stoichiometry_matrix,
    y_min,
    y_max,
    proceed_if_rates_zero,
    rng):
    """
    Construct a complete stochastic stepping class from config classes
    """

    # ============================================================
    # EXACT FAMILY
    # ============================================================
    if isinstance(config, ExactConfig):
        method_cfg = config.method

        if isinstance(method_cfg, DirectMethodConfig):
            return DirectReaction(
                event_rates=event_rates,
                stoichiometry_matrix=stoichiometry_matrix,
                y_min=y_min,
                y_max=y_max,
                proceed_if_rates_zero=proceed_if_rates_zero,
                rng=rng
            )

        elif isinstance(method_cfg, FirstReactionMethodConfig):
            return FirstReaction(
                event_rates=event_rates,
                stoichiometry_matrix=stoichiometry_matrix,
                y_min=y_min,
                y_max=y_max,
                proceed_if_rates_zero=proceed_if_rates_zero,
                rng=rng
            )

        else:
            raise TypeError(f"Unknown Exact family method: {type(method_cfg)}")

    # ============================================================
    # TAU FAMILY
    # ============================================================
    elif isinstance(config, TauConfig):
        # ------------------------
        # Build tau method
        # ------------------------
        method_cfg = config.method

        if isinstance(method_cfg, FixedTauConfig):
            tau_method = Fixed(
                event_rates=event_rates,
                stoichiometry_matrix=stoichiometry_matrix,
                tau=method_cfg.tau,
            )

        elif isinstance(method_cfg, Cao2006TauConfig):

            tau_method = Cao2006(
                event_rates=event_rates,
                stoichiometry_matrix=stoichiometry_matrix,
                transition_mean_func=method_cfg.transition_mean_func,
                transition_var_func=method_cfg.transition_var_func,
                epsilon=method_cfg.epsilon,
            )

        elif isinstance(method_cfg, UKHSA2026TauConfig):

            tau_method = UKHSA2026(
                event_rates=event_rates,
                stoichiometry_matrix=stoichiometry_matrix,
                timestep_mean_func=method_cfg.timestep_mean_func,
                timestep_var_func=method_cfg.timestep_var_func,
                epsilon=method_cfg.epsilon,
            )

        else:
            raise TypeError(f"Unknown Tau method family: {type(method_cfg)}")

        # ------------------------
        # Build checker
        # ------------------------
        checker_cfg = config.checker

        if isinstance(checker_cfg, CriticalReactionConfig):
            checker = CriticalReactionCheck()

        elif isinstance(checker_cfg, ForbiddenStateConfig):
            checker = ForbiddenStateCheck()

        elif isinstance(checker_cfg, NoCheckConfig):
            checker = NoCheck()

        else:
            raise TypeError(f"Unknown checker config: {type(checker_cfg)}")

        # ------------------------
        # Build refiner
        # ------------------------
        refiner_cfg = config.refiner

        if isinstance(refiner_cfg, ProbabilisticRefinerConfig):
            refiner = ProbabilisticTauPrecaution(
                max_retries=refiner_cfg.max_retries,
                acceptable_prob_misstep=refiner_cfg.acceptable_prob_misstep,
                factor_min=refiner_cfg.factor_min,
                factor_max=refiner_cfg.factor_max,
            )

        elif isinstance(refiner_cfg, NoRefinerConfig):
            refiner = NoTauPrecaution()

        else:
            raise TypeError(f"Unknown refiner config: {type(refiner_cfg)}")

        # ------------------------
        # Construct TauLeap
        # ------------------------
        return TauLeap(
            event_rates=event_rates,
            stoichiometry_matrix=stoichiometry_matrix,
            y_min=y_min,
            y_max=y_max,
            proceed_if_rates_zero=proceed_if_rates_zero,
            retry_max=config.retry_max,
            tau_rescale=config.tau_rescale,
            tau_method=tau_method,
            proposal_checker=checker,
            tau_refiner=refiner,
            rng=rng
        )

    # ============================================================
    # Unknown config
    # ============================================================
    else:
        raise TypeError(f"Unknown solver config type: {type(config)}")
