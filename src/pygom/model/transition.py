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

class Event:
    '''
    Class to contain transitions
    '''

    def __init__(self,
                 transition_list,
                 rate=None):
        
        # If one solitary unlisted transition provided, gather it into a list
        if not isinstance(transition_list, list):
            if isinstance(transition_list, Transition):
                transition_list=[transition_list]
            else:
                raise InputStateError("Transition object provided not of class Transition")
        
        # Check each transition is of the type, Transition
        for transition in transition_list:
            if not isinstance(transition, Transition):
                raise InputStateError("At least one Transition object provided not of class Transition")

        # Check that ODE's have not been supplied
        for transition in transition_list:
            if transition.transition_type == TransitionType.ODE:
                raise InputStateError("ODEs cannot be wrapped in an Event class. To pass pure ODEs to"+
                                      "SimulateOde, use the ode argument")

        # Check enough information has been provided and unpack it.
        # Sufficient input consists of either:
        # 1) One transition with an equation + no rate
        # 2) One transition without an equation + rate
        # 3) Multiple transitions each without equations + rate
        # 4) Multiple transitions where only 1 has equation + no rate

        if len(transition_list)==1:
            if transition.equation is not None:
                if rate is not None:
                    raise InputStateError("Event rate dictates the rate at which its"+
                                          "member Transitions occur. It is superfluous for transitions"+
                                          "to declare their own rate and would be incorrect if it differed"+
                                          "from that in the Event anyway.")
                else:
                    self.rate=transition.equation
            elif rate is None:
                raise InputStateError("Rate cannot be found in Event or Transitions")
            else:
                self.rate=rate
        else:
            n_eq=0
            for transition in transition_list:
                if transition.equation is not None:
                    n_eq+=1
            if n_eq>1:
                raise InputStateError("Zero or one equations needed, but ", n_eq, " provided")
            elif (n_eq==1) and (rate is not None):
                raise InputStateError("Rate and equation defined, but only one should be provided")
            elif (n_eq==0) and (rate is None):
                raise InputStateError("Rate cannot be found in Event or Transitions")
            else:
                self.rate=rate
                
        self.transition_list=transition_list
            

                    

class Transition:
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

    def __init__(self,
                 origin=None,
                 equation=None,                 # If equation is given
                 transition_type='ODE',
                 destination=None,
                 magnitude='1',
                 ID=None,
                 name=None):
        '''
        Constructor for the class.

        '''
        self.ID = ID
        self.name = name
        self._setTransitionType(transition_type)
        self._setMagnitude(magnitude)   

        # Check origins and destinations are consistent with transition type

        if self.transition_type == TransitionType.ODE:
            if destination is not None:
                raise InputStateError("Please define ODEs with the dependant variable as the origin")
            if origin is None:
                raise InputStateError("Please define ODEs with the dependant variable as the origin")          
            self._setOrigState(origin)

        if self.transition_type == TransitionType.B:
            if origin is not None:
                # TODO: This warning can be really annoying, I want it to just appear once.
                # print("Update: In the latest version, you should define births as having a destination state instead of an origin.")
                destination=origin
            elif destination is None:
                raise InputStateError("Birth process has no origin or destination")
            # if destination is None:
            #     raise InputStateError("Birth process requires destination")
            # if origin is not None:
            #     raise InputStateError("Birth process can only have a destination, please remove origin")
            self._setDestState(destination)

        if self.transition_type == TransitionType.D:
            if origin is None:
                raise InputStateError("Death process requires origin")
            if destination is not None:
                raise InputStateError("Death process can only have an origin, please remove destination")
            self._setOrigState(origin)

        if self.transition_type == TransitionType.T:
            if origin is None:
                if destination is not None:
                    raise InputStateError("No origin, but transition type is between 2 compartments")
                else:
                    raise InputStateError("No origin or destination, but transition type is between 2 compartments")
            if destination is None:
                raise InputStateError("No destination, but transition type is between 2 compartments")
            if origin == destination:
                raise InputStateError("Origin and destination cannot be the same")
            self._setOrigState(origin)
            self._setDestState(destination)

        # If we get this far, then origin/destination checks have been successful
        # Now add equation if it has been provided

        self._setEquation(equation)
        # if equation is not None:
        #     # TODO: We need to create an event class somehow.
        #     self._setEquation(equation)

    def __str__(self):
        if self.transition_type == TransitionType.T:
            return 'Transition of size %s from %s to %s' % (self._magnitude, self._orig_state, self._dest_state)
        elif self.transition_type == TransitionType.ODE:
            return 'ODE for %s, %s' % (self._orig_state, self._equation)
        elif self.transition_type == TransitionType.B:
            return 'Birth process of size %s into %s' % (self._magnitude, self._orig_state, self._equation)
        elif self.transition_type == TransitionType.D:
            return 'Death process of size %s from %s' % (self._magnitude, self._orig_state, self._equation)

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
    def stochastic(self):
        '''
        Return the secondary effects

        Returns
        -------
        string
            The destination state

        '''
        return self._stochastic

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

    def _setMagnitude(self, magnitude):
        """
        Set the magnitude
        """
        self._magnitude = magnitude
        return self 
    
    def _setStochastic(self, stochastic):
        """
        Set the destination state
        :param destination: Destination State
        :type destination: String
        """
        self._stochastic = stochastic
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
        '''
        Set the transition type
        '''
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
