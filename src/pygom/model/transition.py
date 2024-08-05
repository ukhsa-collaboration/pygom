"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    All classes required to define a transition that is inserted into
    the ode model

"""

__all__ = [
    'Transition',
    'TransitionType'
    ]

from enum import Enum

class TransitionTypeError(Exception):
    '''
    Error when an unknown transition type is inserted
    '''
    pass

class InputStateError(Exception):
    '''
    Error when the input states do not conform to the transition type
    '''
    pass

class Transition(object):
    '''
    This class carries the information for transitions defined
    for an ode, which includes the ode itself, a birth death
    process where only one state is involved and also a transition
    between two states

    Parameters
    ----------
    origin: str
        Origin state.
    equation: str
        Equation defining the transition
    transition_type: enum or str, optional
        of type :class:`TransitionType` or one of ('ODE', 'T', 'B', 'D')
        defaults to 'ODE'
    destination: str, optional
        Destination State.  If the transition is not between state,
        such as a birth or death process, then this is is not
        required.  If it is stated as a birth, death or an ode then
        it throws an error
    '''

    def __init__(self, origin, equation, transition_type='ODE',
                 destination=None, ID=None, name=None):
        '''
        Constructor for the class.

        '''
        self.ID = ID
        self.name = name
        # we naturally assume that the between state transition
        # is false, i.e. everything is either an ode or a birth
        # death process type _equation
        self._betweenStateTransition = False
        self._transition_type = None
        self._setTransitionType(transition_type)

        # private variables
        self._orig_state = None
        self._dest_state = None
        self._equation = None

        if destination is not None:
            if origin == destination:
                if self.transition_type != TransitionType.T:
                    self._setOrigState(origin)
                    self._setEquation(equation)
                else:
                    raise InputStateError("Input have the same state for " +
                                          "the origin and destination, but " +
                                          "transition type is " +
                                          self._transition_type.name)
            else:
                if self.transition_type == TransitionType.T:
                    self._setOrigState(origin)
                    self._setDestState(destination)
                    self._setEquation(equation)
                else:
                    raise InputStateError("Input have both origin and " +
                                          "destination state but transition " +
                                          "type is " + self._transition_type.name)
        else: # no destination
            if self.transition_type != TransitionType.T:
                self._setOrigState(origin)
                self._setEquation(equation)
            else:
                raise InputStateError("Input only have origin, but " +
                                      "transition type is " +
                                      self._transition_type.name)

    def __str__(self):
        if self.transition_type == TransitionType.T:
            return 'Transition from %s to %s, %s' % \
                (self._orig_state, self._dest_state, self._equation)
        elif self.transition_type == TransitionType.ODE:
            return 'ODE for %s, %s' % (self._orig_state, self._equation)
        elif self.transition_type == TransitionType.B:
            return 'Birth process to %s, %s' % (self._orig_state, self._equation)
        elif self.transition_type == TransitionType.D:
            return 'Death process from %s, %s' % (self._orig_state, self._equation)

    def __repr__(self):

        if self.transition_type == TransitionType.T:
            repr_str = """Transition('%s', '%s', 'T', '%s'""" % \
                      (self._orig_state, self._equation, self._dest_state)
        elif self.transition_type == TransitionType.ODE:
            repr_str = """Transition('%s', '%s', 'ODE'""" % \
                      (self._orig_state, self._equation)
        elif self.transition_type == TransitionType.B:
            repr_str = """Transition('%s', '%s', 'B'""" % \
                      (self._orig_state, self._equation)
        elif self.transition_type == TransitionType.D:
            repr_str = """Transition('%s', '%s', 'D'""" % \
                      (self._orig_state, self._equation)

        return repr_str + ", %s, %s)" % (self.ID, self.name)

    def __eq__(self, other):
        if isinstance(other, Transition):
            return self.origin == other.origin and \
            self.destination == other.destination and \
            self.equation == other.equation and \
            self.transition_type == other.transition_type
        else:
            raise NotImplementedError("Can only compare against a " +
                                      "Transition object")

    def __neq__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __le__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __gt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __ge__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    @property
    def origin(self):
        '''
        Return the origin state

        Returns
        -------
        string
            The origin state

        '''
        return self._orig_state

    @property
    def destination(self):
        '''
        Return the destination state

        Returns
        -------
        string
            The destination state

        '''
        return self._dest_state

    @property
    def equation(self):
        '''
        Return the transition _equation

        Returns
        -------
        string
            The transition _equation

        '''
        return self._equation

    @property
    def transition_type(self):
        """
        Return the type of transition

        Returns
        -------
        :class:`.transition_type`
            One of the four type available from :class:`.transition_type`

        """
        return self._transition_type

    def is_between_state(self):
        """
        Return whether it is a transition between two state

        Returns
        -------
        bool
            True if it is a transition between two state
            False if it is only related to the origin state

        """
        return self._transition_type == TransitionType.T

    #
    # Here, we try to put another layer of protection into our code
    # because we only want the parameters of the class to be set at
    # initialization but not after that
    #

    def _setOrigState(self, orig_state):
        """
        Set the original state

        :param origin: Origin State
        :type origin: String
        """
        self._orig_state = orig_state
        return self

    def _setDestState(self, dest_state):
        """
        Set the destination state
        :param destination: Destination State
        :type destination: String
        """
        self._dest_state = dest_state
        return self

    def _setEquation(self, equation):
        '''
        Set the transition _equation
        :param _equation: Transition _equation
        :type _equation: String
        '''
        self._equation = equation
        return self

    def _setTransitionType(self, transition_type):
        # we also need the transition type
        if isinstance(transition_type, TransitionType):
            self._transition_type = transition_type
        elif isinstance(transition_type, str):
            if transition_type.lower() in ('t', 'between states'):
                self._transition_type = TransitionType.T
            elif transition_type.lower() in ('ode', 'ode equation'):
                self._transition_type = TransitionType.ODE
            elif transition_type.lower() in ('b', 'birth process'):
                self._transition_type = TransitionType.B
            elif transition_type.lower() in ('d', 'death process'):
                self._transition_type = TransitionType.D
            else:
                raise TransitionTypeError("Unknown input string, require one" +
                                          " of (T, ODE, D, B)")
        else:
            raise TransitionTypeError("Input transitionType requires a " +
                                      "TransitionType object or str")

class TransitionType(Enum):
    '''
    This is an Enum describing the four feasible type of transitions use to
    define the ode model :class:`BaseOdeModel`

    The following four types of transitions are available.

    B = Birth process

    D = Death process

    T = Transition between states

    ODE = ODE _equation

    '''
    B = 'Birth process'
    D = 'Death process'
    T = 'Between states'
    ODE = 'ODE _equation'
