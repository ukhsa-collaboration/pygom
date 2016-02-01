"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    All classes required to define a transition that is inserted into the ode model

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
    process where only one state is involved and also a getEquation
    between two states

    Parameters
    ----------
    origState: string
        Origin state.
    equation: string
        Equation defining the transition
    transitionType: enum
        of type :class:`TransitionType`
    destState: string, optional
        Destination State.  If the transition is not between state,
        such as a birth or death process, then this is is not
        required.  If it is stated as a birth, death or an ode then
        it throws an error
    '''

    def __init__(self, origState, equation, transitionType, destState=None):
        '''
        Constructor for the class.

        '''
        # we naturally assume that the between state getEquation
        # is false, i.e. everything is either an ode or a birth
        # death process type _equation
        self._betweenStateTransition = False

        # we also need the getEquation type
        if isinstance(transitionType, TransitionType):
            self._transitionType = transitionType
        elif isinstance(transitionType, str):
            if transitionType == 'T':
                self._transitionType = TransitionType.T
            elif transitionType == 'ODE':
                self._transitionType = TransitionType.ODE
            elif transitionType == 'B':
                self._transitionType = TransitionType.B
            elif transitionType == 'D':
                self._transitionType = TransitionType.D
            else:
                raise TransitionTypeError("Unknown input string, require one of T,ODE,D,B")
        else:
            raise TransitionTypeError("Input getEquation type requires a TransitionType object")

        # private variables
        self._origState = None
        self._destState = None
        self._equation = None

        if destState is not None:
            if origState == destState:
                if self._transitionType != TransitionType.T:
                    self.__setOrigState(origState)
                    self.__setEquation(equation)
                else:
                    raise InputStateError("Input have the same state for the origin "
                                          +"and destination, but transition type is "
                                          +self._transitionType.name)
            else:
                if self._transitionType == TransitionType.T:
                    self.__setOrigState(origState)
                    self.__setDestState(destState)
                    self.__setEquation(equation)
                else:
                    raise InputStateError("Input have both origin and destination state "
                                          +"but transition type is "+self._transitionType.name)
        else: # no destination
            if self._transitionType != TransitionType.T:
                self.__setOrigState(origState)
                self.__setEquation(equation)
            else:
                raise InputStateError("Input only have origin, but transition type is "
                                      +self._transitionType.name)

    def getOrigState(self):
        '''
        Return the origin state

        Returns
        -------
        string
            The origin state

        '''
        return self._origState

    def getDestState(self):
        '''
        Return the destination state

        Returns
        -------
        string
            The destination state

        '''
        return self._destState

    def getEquation(self):
        '''
        Return the transition getEquation

        Returns
        -------
        string
            The transition _equation

        '''
        return self._equation

    def getTransitionType(self):
        """
        Return the type of transition

        Returns
        -------
        :class:`.getTransitionType`
            One of the four type available from :class:`.getTransitionType`

        """
        return self._transitionType

    def getBetweenStateTransition(self):
        """
        Return whether it is a transition between two state

        Returns
        -------
        bool
            True if it is a transition between two state
            False if it is only related to the origin state

        """
        if self._transitionType == TransitionType.T:
            return True
        else:
            return False

    #
    # Here, we try to put another layer of protection into our code
    # because we only want the parameters of the class to be set at
    # initialization but not after that
    #

    def __setOrigState(self, origState):
        """
        Set the original state

        :param getOrigState: Origin State
        :type getOrigState: String
        """
        self._origState = origState
        return None

    def __setDestState(self, destState):
        """
        Set the destination state
        :param getDestState: Destination State
        :type getDestState: String
        """
        self._destState = destState
        return None

    def __setEquation(self, equation):
        '''
        Set the transition getEquation
        :param getEquation: Transition getEquation
        :type getEquation: String
        '''
        self._equation = equation
        return None

class TransitionType(Enum):
    '''
    This is an Enum describing the four feasible type of transitions use to
    define the ode model :class:`operateOdeModel`

    The following four types of transitions are available.

    B = Birth process

    D = Death process

    T = Transition between states

    ODE = ODE getEquation

    '''
    B = 'Birth process'
    D = 'Death process'
    T = 'Transition between states'
    ODE = 'ODE getEquation'
