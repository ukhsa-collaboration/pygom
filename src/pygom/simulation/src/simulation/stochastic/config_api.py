"""
Build config data classes from user input
"""

import warnings

from typing import Any, Dict, Union

from .config import (
    SolverConfig,
    ExactConfig, TauConfig,
    ExactMethodConfig, TauMethodConfig,
    DirectMethodConfig, FirstReactionMethodConfig,
    FixedTauConfig, Cao2006TauConfig, UKHSA2026TauConfig,
    CheckerConfig, CriticalReactionConfig, ForbiddenStateConfig, NoCheckConfig,
    TauRefinerConfig, ProbabilisticRefinerConfig, NoRefinerConfig,
)

MethodLike = Union[str, ExactMethodConfig, TauMethodConfig]
CheckerLike = Union[str, CheckerConfig]
RefinerLike = Union[str, TauRefinerConfig]

def _build_checker(spec: CheckerLike, **opts: Any) -> CheckerConfig:
    """
    Build tau leap step checker
    """
    if isinstance(spec, CheckerConfig):
        return spec

    name = str(spec).lower()
    if name == "forbidden_reaction":
        return CriticalReactionConfig()
    if name == "forbidden_state":
        return ForbiddenStateConfig()
    if name == "none":
        return NoCheckConfig()

    raise ValueError(
        f"Unknown checker spec: {spec!r}. Known: 'forbidden_reaction', 'forbidden_state', 'none'"
    )

def _build_refiner(spec: RefinerLike, **opts: Any) -> TauRefinerConfig:
    """
    Build tau leap precautionary step size refiner
    """
    if isinstance(spec, TauRefinerConfig):
        return spec

    name = str(spec).lower()
    if name == "prob":
        allowed = {"max_retries", "acceptable_prob_misstep", "factor_min", "factor_max"}
        # only pass supported kwargs to the dataclass
        filtered = {k: v for k, v in opts.items() if k in allowed}
        return ProbabilisticRefinerConfig(**filtered)
    if name == "none":
        return NoRefinerConfig()

    raise ValueError(
        f"Unknown refiner spec: {spec!r}. Known: 'prob', 'none'"
    )

def _method_from_string(name: str, **options: Any) -> Union[ExactMethodConfig, TauMethodConfig]:
    """
    Translate string method names to their corresponding method config dataclass.
    Raise error if required options are missing.
    """
    key = name.lower()

    # ===== exact family =====
    if key == "direct":
        unused = list(options)
        if unused:
            warnings.warn(f"method='direct' does not use variables: {', '.join(unused)}", RuntimeWarning)
        return DirectMethodConfig()
    
    if key == "first_reaction":
        unused = list(options)
        if unused:
            warnings.warn(f"method='first_reaction' does not use variables: {', '.join(unused)}", RuntimeWarning)
        return FirstReactionMethodConfig()

    # ===== tau family =====
    if key == "fixed_tau":        
        required = ["tau"]
        optional = ["checker", "checker_opts", "refiner", "refiner_opts"]
        missing = [name for name in required if name not in options]
        if missing:
            raise ValueError(f"method='fixed_tau' missing required options: {', '.join(missing)}")

        unused = [name for name in options if name not in (required+optional)]
        if unused:
            warnings.warn(f"method='fixed_tau' does not use variables: {', '.join(unused)}", RuntimeWarning)

        return FixedTauConfig(tau=float(options["tau"]))

    if key == "cao2006":
        required = ["epsilon", "transition_mean_func", "transition_var_func"]
        optional = ["checker", "checker_opts", "refiner", "refiner_opts"]
        missing = [name for name in required if name not in options]
        if missing:
            raise ValueError(f"method='cao2006' missing required options: {', '.join(missing)}")
        
        unused = [name for name in options if name not in (required+optional)]
        if unused:
            warnings.warn(f"method='cao2006' does not use variables: {', '.join(unused)}", RuntimeWarning)

        return Cao2006TauConfig(
            epsilon=float(options["epsilon"]),
            transition_mean_func=options["transition_mean_func"],
            transition_var_func=options["transition_var_func"]
        )

    if key == "ukhsa2026":
        required = ["epsilon", "timestep_mean_func", "timestep_var_func"]
        optional = ["checker", "checker_opts", "refiner", "refiner_opts"]
        missing = [name for name in required if name not in options]
        if missing:
            raise ValueError(f"method='ukhsa2026' missing required options: {', '.join(missing)}")
        
        unused = [name for name in options if name not in (required+optional)]
        if unused:
            warnings.warn(f"method='ukhsa2026' does not use variables: {', '.join(unused)}", RuntimeWarning)

        return UKHSA2026TauConfig(
            epsilon=float(options["epsilon"]),
            timestep_mean_func=options["timestep_mean_func"],
            timestep_var_func=options["timestep_var_func"]
        )

    raise ValueError(
        f"Unknown method {name!r}. "
        "Known: 'direct', 'first_reaction', 'fixed_tau', 'cao2006', 'ukhsa2026'"
    )


def build_config(method: MethodLike, /, **options: Any) -> SolverConfig:
    """
    Build a high-level SolverConfig from a solve_ivp-like interface.

    Parameters
    ----------
    method : str | ExactMethodConfig | TauMethodConfig
        - String aliases:
            Exact: 'direct', 'first_reaction'
            Tau:   'fixed_tau', 'cao2006', 'ukhsa2026'
        - Or pass a method config object directly, e.g. FixedTauConfig(tau=0.1)

    **options : dict
        For Tau methods:
            checker: str | CheckerConfig = 'critical'
            checker_opts: dict = {}
            refiner: str | TauRefinerConfig = 'none'
            refiner_opts: dict = {}
            retry_max: int = 10
            tau_rescale: float = 0.5

        For method-specific strings:
            - fixed_tau: tau: float (required)
            - cao2006: epsilon: float (required)
            - ukhsa2026: epsilon: float (required)

        If you pass a method config object, options specific to that method are ignored.

    Returns
    -------
    SolverConfig
        Either ExactConfig or TauConfig depending on the method.
    """
    # If the user passed a config object, we can infer the family directly
    if isinstance(method, ExactMethodConfig):
        return ExactConfig(method=method)

    if isinstance(method, TauMethodConfig):
        checker_spec = options.pop("checker", "forbidden_reaction")
        checker_opts: Dict[str, Any] = options.pop("checker_opts", {}) or {}
        refiner_spec = options.pop("refiner", "none")
        refiner_opts: Dict[str, Any] = options.pop("refiner_opts", {}) or {}
        retry_max = int(options.pop("retry_max", 10))
        tau_rescale = float(options.pop("tau_rescale", 0.5))

        checker = _build_checker(checker_spec, **checker_opts)
        refiner = _build_refiner(refiner_spec, **refiner_opts)

        return TauConfig(
            method=method,
            checker=checker,
            refiner=refiner,
            retry_max=retry_max,
            tau_rescale=tau_rescale,
        )

    # Otherwise treat it as a string method name
    if isinstance(method, str):
        method_cfg = _method_from_string(method, **options)

        if isinstance(method_cfg, ExactMethodConfig):
            return ExactConfig(method=method_cfg)

        elif isinstance(method_cfg, TauMethodConfig):
            # pull tau-level options (do not consume method-specific ones here)
            checker_spec = options.pop("checker", "forbidden_reaction")
            checker_opts: Dict[str, Any] = options.pop("checker_opts", {}) or {}
            refiner_spec = options.pop("refiner", "none")
            refiner_opts: Dict[str, Any] = options.pop("refiner_opts", {}) or {}
            retry_max = int(options.pop("retry_max", 10))
            tau_rescale = float(options.pop("tau_rescale", 0.5))

            checker = _build_checker(checker_spec, **checker_opts)
            refiner = _build_refiner(refiner_spec, **refiner_opts)

            return TauConfig(
                method=method_cfg,
                checker=checker,
                refiner=refiner,
                retry_max=retry_max,
                tau_rescale=tau_rescale,
            )

    raise TypeError(
        "method must be a string alias or a method-config instance "
        f"(got {type(method).__name__})."
    )