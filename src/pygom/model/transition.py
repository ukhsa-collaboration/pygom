"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    All classes required to define a transition that is inserted into
    the ode model

"""

__all__ = [
    'Event',
    'Transition',
    'TransitionType'
]

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet

class TransitionTypeError(Exception):
    """Error when an unknown transition type is inserted"""
    pass

class InputStateError(Exception):
    """Error when the input states do not conform to the transition type"""
    pass

@dataclass(frozen=True)
class TransitionSpec:
    description: str
    required: FrozenSet[str]

class TransitionType(Enum):
    B = TransitionSpec(
        description="Birth process",
        required=frozenset({"destination"}),
    )

    D = TransitionSpec(
        description="Death process",
        required=frozenset({"origin"}),
    )

    T = TransitionSpec(
        description="Between states",
        required=frozenset({"origin", "destination"}),
    )

    @property
    def description(self) -> str:
        return self.value.description

    @property
    def required(self) -> FrozenSet:
        return self.value.required

    @property
    def forbidden(self) -> set:
        return {"origin", "destination"} - self.required

    @classmethod
    def from_string(cls, value: str) -> "TransitionType":
        value = value.lower()

        if value in ("b", "birth process"):
            return cls.B

        if value in ("d", "death process"):
            return cls.D

        if value in ("t", "between states"):
            return cls.T

        raise TransitionTypeError(
            "Unknown input string, require one of (T, D, B)"
        )

def infer_transition_type(origin, destination) -> TransitionType:
    """
    Determine the transition type purely from the
    origin/destination specification.
    """
    if (origin is None) and (destination is not None):
        return TransitionType.B

    if (origin is not None) and (destination is None):
        return TransitionType.D

    if (origin is not None) and (destination is not None):
        return TransitionType.T

    raise InputStateError(
        "No origin or destination state provided."
    )

@dataclass
class Transition:
    origin: str | None = None
    destination: str | None = None
    transition_type: TransitionType | str | None = None
    magnitude: str = "1"
    ID: int | None = None
    name: str | None = None
    rate: str | None = None

    def __post_init__(self):

        # Allow strings as well as enum values
        if isinstance(self.transition_type, str):
            self.transition_type = TransitionType.from_string(self.transition_type)

        inferred_type = infer_transition_type(self.origin, self.destination)

        # User did not specify a type: use the inferred one.
        if self.transition_type is None:
            self.transition_type = inferred_type

        # User specified a type: verify consistency.
        elif self.transition_type != inferred_type:
            raise InputStateError(
                f"Specified transition type "
                f"{self.transition_type.name} is inconsistent "
                f"with origin={self.origin!r}, "
                f"destination={self.destination!r}. "
                f"Inferred type is {inferred_type.name}."
            )

        # Additional modelling rules
        if (
            self.transition_type is TransitionType.T
            and self.origin == self.destination
        ):
            raise InputStateError(
                "Origin and destination cannot be the same."
            )

    @property
    def is_between_state(self) -> bool:
        return self.transition_type is TransitionType.T

    def __str__(self):
        if self.transition_type is TransitionType.B:
            return (
                f"Birth process of size {self.magnitude} "
                f"into {self.destination}"
            )

        if self.transition_type is TransitionType.D:
            return (
                f"Death process of size {self.magnitude} "
                f"from {self.origin}"
            )

        return (
            f"Transition of size {self.magnitude} "
            f"from {self.origin} "
            f"to {self.destination}"
        )

@dataclass
class Event:
    rate: str
    transition_list: list[Transition]
    ID: int | None = None

    # @classmethod
    # def single_transition(cls, rate, **kwargs):
    #     return cls(
    #         rate=rate,
    #         transition_list=[Transition(**kwargs)]
    #     )