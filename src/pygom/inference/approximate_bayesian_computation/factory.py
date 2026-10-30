"""
Factory to construct distance and covariance classes from user input
"""

# TODO: allow (likely advanced) users to pass classes directly

from dataclasses import dataclass
import warnings

from .particle_distance import Distance, EuclidDistance, ScaledDistance, MahalanobisDistance
from .covariance import CovarianceMethod, GlobalCovariance, LocalCovariance, KNNCovariance

def validate_options(SPECS, module, method, kwargs):
    """
    Raise error if required arguments are missing and warning
    if unused arguments are supplied
    """

    spec = SPECS[method]

    supplied = {
        name
        for name, value in kwargs.items()
        if value is not None
    }

    missing = spec.required - supplied
    if missing:
        raise TypeError(f"{module}='{method}' requires {sorted(missing)}")

    unused = supplied - spec.required - spec.optional
    if unused:
        warnings.warn(f"Arguments {sorted(unused)} ignored for {module}='{method}'")

@dataclass
class MethodSpec:
    required: set
    optional: set

#############################################
# Distance
#############################################

DISTANCE_SPECS = {
    "euclidean": MethodSpec(
        required=set(),
        optional=set()
    ),
    "scaled": MethodSpec(
        required=set(),
        optional={"lengthscales"}
    ),
    "mahalanobis": MethodSpec(
        required=set(),
        optional={"regularisation"}
    )
}

def build_distance(
    distance_method,
    *,
    lengthscales=None,
    regularisation=None
):

    validate_options(DISTANCE_SPECS, "distance", distance_method, {"lengthscales": lengthscales, "regularisation": regularisation})

    if distance_method == "euclidean":
        return EuclidDistance()

    if distance_method == "scaled":
        return ScaledDistance(
            lengthscales=lengthscales
        )

    if distance_method == "mahalanobis":

        if regularisation is None:
            regularisation = 1e-10

        return MahalanobisDistance(
            regularisation=regularisation
        )

    raise ValueError(
        f"Unknown distance '{distance_method}'"
    )

#############################################
# COVARIANCE
#############################################

COVARIANCE_SPECS = {
    "global": MethodSpec(
        required=set(),
        optional=set()
    ),
    "local": MethodSpec(
        required=set(),
        optional=set()
    ),
    "knn": MethodSpec(
        required={"k", "distance"},
        optional=set()
    )
}

def build_covariance(
    covariance_method,
    *,
    k=None,
    distance_method=None,
    lengthscales=None,
    regularisation=None
):

    validate_options(COVARIANCE_SPECS, "covariance", covariance_method, {"k": k, "distance": distance_method})

    if covariance_method == "global":
        return GlobalCovariance()

    if covariance_method == "local":
        return LocalCovariance()

    if covariance_method == "knn":

        distance_method = build_distance(
            distance_method=distance_method,
            lengthscales=lengthscales,
            regularisation=regularisation
        )

        return KNNCovariance(
            k=k,
            distance_method=distance_method,
        )

    raise ValueError(
        f"Unknown covariance '{covariance_method}'"
    )