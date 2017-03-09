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
    origState: str
        Origin state.
    equation: str
        Equation defining the transition
    transitionType: enum or str, optional
        of type :class:`TransitionType` or one of ('ODE', 'T', 'B', 'D')
        defaults to 'ODE'
    destState: str, optional
        Destination State.  If the transition is not between state,
        such as a birth or death process, then this is is not
        required.  If it is stated as a birth, death or an ode then
        it throws an error
    '''

    def __init__(self, origState, equation, transitionType='ODE',
                 destState=None, ID=None, name=None):
        '''
        Constructor for the class.

        '''
        self.ID = ID
        self.name = name
        # we naturally assume that the between state transition
        # is false, i.e. everything is either an ode or a birth
        # death process type _equation
        self._betweenStateTransition = False

        # we also need the transition type
        if isinstance(transitionType, TransitionType):
            self.transitionType = transitionType
        elif isinstance(transitionType, str):
            if transitionType.lower() in ('t', 'between states'):
                self.transitionType = TransitionType.T
            elif transitionType.lower() in ('ode', 'ode equation'):
                self.transitionType = TransitionType.ODE
            elif transitionType.lower() in ('b', 'birth process'):
                self.transitionType = TransitionType.B
            elif transitionType.lower() in ('d', 'death process'):
                self.transitionType = TransitionType.D
            else:
                raise TransitionTypeError("Unknown input string, require one " + 
                                          "of (T, ODE, D, B)")
        else:
            raise TransitionTypeError("Input transitionType requires a " + 
                                      "TransitionType object or str")

        # private variables
        self.origState = None
        self.destState = None
        self.equation = None

        if destState is not None:
            if origState == destState:
                if self.transitionType != TransitionType.T:
                    self.__setOrigState(origState)
                    self.__setEquation(equation)
                else:
                    raise InputStateError("Input have the same state for the " +
                                          "origin and destination, but " + 
                                          "transition type is " + 
                                          self.transitionType.name)
            else:
                if self.transitionType == TransitionType.T:
                    self.__setOrigState(origState)
                    self.__setDestState(destState)
                    self.__setEquation(equation)
                else:
                    raise InputStateError("Input have both origin and " +
                                          "destination state but transition" +
                                          "type is " + self.transitionType.name)
        else: # no destination
            if self.transitionType != TransitionType.T:
                self.__setOrigState(origState)
                self.__setEquation(equation)
            else:
                raise InputStateError("Input only have origin, but " + 
                                      "transition type is " + 
                                      self.transitionType.name)
                
    def __str__(self):
        if self.transitionType == TransitionType.T:
            return 'Transition from %s to %s, %s' % \
                (self.origState, self.destState, self.equation)
        elif self.transitionType == TransitionType.ODE:
            return 'ODE for %s, %s' % (self.origState, self.equation)
        elif self.transitionType == TransitionType.B:
            return 'Birth process to %s, %s' % (self.origState, self.equation)
        elif self.transitionType == TransitionType.D:
            return 'Death process from %s, %s' % (self.origState, self.equation)
        
    def __repr__(self):
        
        if self.transitionType == TransitionType.T:
            reprStr = """Transition('%s', '%s', 'T', '%s'""" % \
                      (self.origState, self.equation, self.destState)
        elif self.transitionType == TransitionType.ODE:
            reprStr = """Transition('%s', '%s', 'ODE'""" % \
                      (self.origState, self.equation)
        elif self.transitionType == TransitionType.B:
            reprStr = """Transition('%s', '%s', 'B'""" % \
                      (self.origState, self.equation)
        elif self.transitionType == TransitionType.D:
            reprStr = """Transition('%s', '%s', 'D'""" % \
                      (self.origState, self.equation)
        
        return reprStr + ", %s, %s)" % (self.ID, self.name)

    def __eq__(self, other):
        if isinstance(other, Transition):
            return self.origState == other.origState and \
            self.destState == other.destState and \
            self.equation == other.equation and \
            self.transitionType == other.transitionType
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
    
    def getOrigState(self):
        '''
        Return the origin state

        Returns
        -------
        string
            The origin state

        '''
        return self.origState

    def getDestState(self):
        '''
        Return the destination state

        Returns
        -------
        string
            The destination state

        '''
        return self.destState

    def getEquation(self):
        '''
        Return the transition equation

        Returns
        -------
        string
            The transition equation

        '''
        return self.equation

    def getTransitionType(self):
        """
        Return the type of transition

        Returns
        -------
        :class:`.getTransitionType`
            One of the four type available from :class:`.getTransitionType`

        """
        return self.transitionType

    def getBetweenStateTransition(self):
        """
        Return whether it is a transition between two state

        Returns
        -------
        bool
            True if it is a transition between two state
            False if it is only related to the origin state

        """
        if self.transitionType == TransitionType.T:
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
        self.origState = origState
        return None

    def __setDestState(self, destState):
        """
        Set the destination state
        :param getDestState: Destination State
        :type getDestState: String
        """
        self.destState = destState
        return None

    def __setEquation(self, equation):
        '''
        Set the transition equation
        :param equation: Transition equation
        :type equation: String
        '''
        self.equation = equation
        return None

class TransitionType(Enum):
    '''
    This is an Enum describing the four feasible type of transitions use to
    define the ode model :class:`BaseOdeModel`

    The following four types of transitions are available.

    B = Birth process

    D = Death process

    T = Transition between states

    ODE = ODE equation

    '''
    B = 'Birth process'
    D = 'Death process'
    T = 'Between states'
    ODE = 'ODE equation'
