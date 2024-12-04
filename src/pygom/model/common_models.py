"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    A set of commonly used models

"""

from collections import OrderedDict

from .transition import TransitionType, Transition, Event
#from .deterministic import DeterministicOde
from .simulate import SimulateOde


def SIS(param=None):
    """
    Susceptible Infected Susceptible model

    .. math::
        \\frac{dS}{dt} &= -\\beta SI + \\gamma I \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SIS({'beta':0.5, 'gamma':0.2})
    >>> t = np.linspace(0, 20, 100)
    >>> x0 = [1.0, 0.1]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    state_list = [('S', (0,None)), ('I', (0, None))]
    param_list = ['beta', 'gamma', 'N']

    trans_inf=Transition(origin='S', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='beta*S*I/N', transition_list=[trans_inf])

    trans_rec=Transition(origin='I', destination='S',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    model = SimulateOde(state=state_list,
                        param=param_list,
                        event=[event_inf, event_rec])

    if param is None:
        return model
    else:
        model.parameters = param
        return model


def SIS_Periodic(param=None):
    """
    A SIS model with periodic contact, defined by the ode as per [Hethcote1973]_

    .. math::
        \\frac{dI}{dt} = (\\beta(t)N - \\alpha) I - \\beta(t)I^{2}

    where

    .. math::
        \\beta(t) = 2 - 1.8 \\cos(5t).

    As the name suggests, it achieves a (stable) periodic solution.

    Examples
    --------
    >>> from pygom import common_models
    >>> import numpy as np
    >>> ode = common_models.SIS_Periodic({'alpha':1.0})
    >>> t = np.linspace(0, 10, 101)
    >>> x0 = [0.1, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    state_list = [('S', (0,None)), ('I', (0, None))]
    param_list = ['gamma', 'beta0', 'delta', 'period', 'N']
    derived_param = [('betaT', 'beta0*(1-delta*cos(2*3.14159*t/period))')]

    trans_inf=Transition(origin='S', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='betaT*S*I/N', transition_list=[trans_inf])

    trans_rec=Transition(origin='I', destination='S',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    model = SimulateOde(state=state_list,
                        param=param_list,
                        derived_param=derived_param,
                        event=[event_inf, event_rec])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SIR(param=None):
    """
    A standard SIR model as per [Brauer2008]_

    .. math::
        \\frac{dS}{dt} &= -\\ \\frac{beta SI}{N} \\\\
        \\frac{dI}{dt} &= \\ \\frac{beta SI}{N} - \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I


    Examples
    --------
    The model that produced top two graph in Figure 1.3 of the reference above.
    First, when everyone is susceptible and only one individual was infected.

    >>> import numpy as np
    >>> from pygom import common_models
    >>> N=1e5
    >>> ode = common_models.SIR({'beta':0.5, 'gamma':0.2, 'N':N})
    >>> t = np.linspace(0, 730, 1001)
    >>> i0=1
    >>> x0 = [N-i0, i0, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    """

    state_list = [('S', (0,None)), ('I', (0, None)), ('R', (0, None))]
    param_list = ['beta', 'gamma', 'N']

    trans_inf=Transition(origin='S', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='beta*S*I/N', transition_list=[trans_inf])

    trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    model = SimulateOde(state=state_list,
                        param=param_list,
                        event=[event_inf, event_rec])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SEIR(param=None, init=None):
    """
    A standard SIR model [Brauer2008]_ with population N

    """

    state_list = [('S', (0,None)), ('E', (0,None)), ('I', (0, None)), ('R', (0, None))]
    param_list = ['beta', 'alpha', 'gamma', 'N']

    trans_exp=Transition(origin='S', destination='E',transition_type=TransitionType.T)
    event_exp=Event(rate='beta*S*I/N', transition_list=[trans_exp])

    trans_inf=Transition(origin='E', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='alpha*E', transition_list=[trans_inf])

    trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    model = SimulateOde(state=state_list,
                        param=param_list,
                        event=[event_exp,
                               event_inf,
                               event_rec])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SIR_Birth_Death(param=None):
    """
    Extension of the standard SIR model [Brauer2008]_ to also include birth and death

    .. math::
        \\frac{dS}{dt} &= B -\\beta SI - \\mu S \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I - \\mu I \\\\
        \\frac{dR}{dt} &= \\gamma I


    Examples
    --------
    The model that produced bottom graph in Figure 1.3 of the reference above.

    >>> import numpy as np
    >>> from pygom import common_models
    >>> B = 126372.0/365.0
    >>> N = 7781984.0
    >>> params = {'beta':3.6, 'gamma':0.2, 'B':B/N, 'mu':B/N}
    >>> ode = common_models.SIR_Birth_Death(params)
    >>> t = np.linspace(0, 35*365, 10001)
    >>> x0 = [0.065, 123.0*(5.0/30.0)/N, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution,output = ode.integrate(t[1::], full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`.SIR`
    """

    state_list = [('S', (0,None)), ('I', (0, None)), ('R', (0, None)), ('N', (0, None))]
    param_list=['beta', 'gamma', 'mu']

    # Infection
    trans_inf=Transition(origin='S', destination='I', transition_type='T')
    event_inf=Event(transition_list=[trans_inf], rate='beta*S*I/N')

    # Recovery
    trans_rec=Transition(origin='I', destination='R', transition_type='T')
    event_rec=Event(transition_list=[trans_rec], rate='gamma*I')

    # Transitions in and out of total population count
    # These will be used in the following event definitions
    birth_N=Transition(destination="N", transition_type="B")
    death_N=Transition(origin="N", transition_type="D")

    # 1) Birth event into S
    birth=Transition(destination="S", transition_type="B")
    event_birth=Event(transition_list=[birth, birth_N], rate='mu*N')

    # 2) Death event of an S
    death_S=Transition(origin="S", transition_type="D")
    event_death_S=Event(transition_list=[death_S, death_N], rate='mu*S')

    # 3) Death event of an I
    death_I=Transition(origin="I", transition_type="D")
    event_death_I=Event(transition_list=[death_I, death_N], rate='mu*I')

    # 4) Death event of an R
    death_R=Transition(origin="R", transition_type="D")
    event_death_R=Event(transition_list=[death_R, death_N], rate='mu*R')


    model=SimulateOde(state=state_list,
                      param=param_list,
                      event=[event_inf,
                             event_rec,
                             event_birth,
                             event_death_S,
                             event_death_I,
                             event_death_R])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SEIR_Birth_Death(param=None):
    """
    A standard SEIR model with birth and death [Aron1984]_, defined by the ode

    .. math::
        \\frac{dS}{dt} &= \\mu - \\beta SI - \\mu S \\\\
        \\frac{dE}{dt} &= \\beta SI - (\\mu + \\alpha) E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\mu + \\gamma) I \\\\
        \\frac{dR}{dt} &= \\gamma I

    Examples
    --------
    Uses the same set of parameters as the examples in :func:`.SEIR`
    apart from :math:`\\mu` which is new.

    >>> import numpy as np
    >>> from pygom import common_models
    >>> params = {'beta':1800, 'gamma':100, 'alpha':35.84, 'mu':0.02}
    >>> ode = common_models.SEIR_Birth_Death(params)
    >>> t = np.linspace(0, 50, 1001)
    >>> x0 = [0.0658, 0.0007, 0.0002, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution,output = ode.integrate(t[1::], full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`.SEIR`
    """

    state_list = [('S', (0,None)), ('E', (0,None)), ('I', (0, None)), ('R', (0, None)), ('N', (0, None))]
    param_list = ['beta', 'alpha', 'gamma', 'mu']

    # Exposure
    trans_exp=Transition(origin='S', destination='E',transition_type=TransitionType.T)
    event_exp=Event(rate='beta*S*I/N', transition_list=[trans_exp])

    # Infectious
    trans_inf=Transition(origin='E', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='alpha*E', transition_list=[trans_inf])

    # Recovery
    trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    # Transitions in and out of total population count
    # These will be used in the following event definitions
    birth_N=Transition(destination="N", transition_type="B")
    death_N=Transition(origin="N", transition_type="D")

    # 1) Birth event into S
    birth=Transition(destination="S", transition_type="B")
    event_birth=Event(transition_list=[birth, birth_N], rate='mu*N')

    # 2) Death event of an S
    death_S=Transition(origin="S", transition_type="D")
    event_death_S=Event(transition_list=[death_S, death_N], rate='mu*S')

    # 3) Death event of an E
    death_E=Transition(origin="E", transition_type="D")
    event_death_E=Event(transition_list=[death_E, death_N], rate='mu*E')

    # 4) Death event of an I
    death_I=Transition(origin="I", transition_type="D")
    event_death_I=Event(transition_list=[death_I, death_N], rate='mu*I')

    # 5) Death event of an R
    death_R=Transition(origin="R", transition_type="D")
    event_death_R=Event(transition_list=[death_R, death_N], rate='mu*R')


    model=SimulateOde(state=state_list,
                      param=param_list,
                      event=[event_exp,
                             event_inf,
                             event_rec,
                             event_birth,
                             event_death_S,
                             event_death_E,
                             event_death_I,
                             event_death_R])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SEIR_Birth_Death_Periodic(param=None):
    """
    A SEIR birth death model with periodic contact [Aron1984]_, defined by the ode

    .. math::
        \\frac{dS}{dt} &= \\mu - \\beta(t)SI - \\mu S \\\\
        \\frac{dE}{dt} &= \\beta(t)SI - (\\mu + \\alpha) E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\mu + \\gamma) I \\\\
        \\frac{dR}{dt} &= \\gamma I

    where

    .. math::
        \\beta(t) = \\beta_{0} (1 + \\beta_{1} \\cos(2 \\pi t)).

    An extension of an SEIR birth death model by varying the contact rate
    through time.

    Examples
    --------
    Uses the same set of parameters as the examples in
    :py:func:`.SEIR_Birth_Death` but now we have two beta parameters instead of one.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from pygom import common_models
    >>> params = {'beta0':1800, 'beta1':0.2, 'gamma':100, 'alpha':35.84, 'mu':0.02}
    >>> ode = common_models.SEIR_Birth_Death_Periodic(params)
    >>> t = np.linspace(0, 50, 1001)
    >>> x0 = [0.0658, 0.0007, 0.0002, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution,output = ode.integrate(t[1::], full_output=True)
    >>> ode.plot()
    >>> plt.plot(np.log(solution[:,0]), np.log(solution[:,1]))
    >>> plt.show()
    >>> plt.plot(np.log(solution[:,0]), np.log(solution[:,2]))
    >>> plt.show()

    See also
    --------
    :func:`.SEIR`, :func:`.SEIR_Birth_Death`, :func:`.SIR_Periodic`

    """
    state_list = [('S', (0,None)), ('E', (0,None)), ('I', (0, None)), ('R', (0, None)), ('N', (0, None))]
    param_list = ['beta0', 'delta', 'period', 'alpha', 'gamma', 'mu']
    derived_param = [('betaT', 'beta0*(1-delta*cos(2*3.14159*t/period))')]

    # Exposure
    trans_exp=Transition(origin='S', destination='E',transition_type=TransitionType.T)
    event_exp=Event(rate='betaT*S*I/N', transition_list=[trans_exp])

    # Infectious
    trans_inf=Transition(origin='E', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='alpha*E', transition_list=[trans_inf])

    # Recovery
    trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    # Transitions in and out of total population count
    # These will be used in the following event definitions
    birth_N=Transition(destination="N", transition_type="B")
    death_N=Transition(origin="N", transition_type="D")

    # 1) Birth event into S
    birth=Transition(destination="S", transition_type="B")
    event_birth=Event(transition_list=[birth, birth_N], rate='mu*N')

    # 2) Death event of an S
    death_S=Transition(origin="S", transition_type="D")
    event_death_S=Event(transition_list=[death_S, death_N], rate='mu*S')

    # 3) Death event of an E
    death_E=Transition(origin="E", transition_type="D")
    event_death_E=Event(transition_list=[death_E, death_N], rate='mu*E')

    # 4) Death event of an I
    death_I=Transition(origin="I", transition_type="D")
    event_death_I=Event(transition_list=[death_I, death_N], rate='mu*I')

    # 5) Death event of an R
    death_R=Transition(origin="R", transition_type="D")
    event_death_R=Event(transition_list=[death_R, death_N], rate='mu*R')

    model=SimulateOde(state=state_list,
                      param=param_list,
                      derived_param=derived_param,
                      event=[event_exp,
                             event_inf,
                             event_rec,
                             event_birth,
                             event_death_S,
                             event_death_E,
                             event_death_I,
                             event_death_R])

    if param is None:
        return model
    else:
        model.parameters = param
        return model


def SEIR_Birth_Death_Periodic_Waning_Intro(param=None):
    '''
    SEIR model with vital dynamics, periodic infectivity, immune waning and
    external introductions
    '''
    state_list = [('S', (0,None)), ('E', (0,None)), ('I', (0, None)), ('R', (0, None)), ('N', (0, None))]
    param_list = ['beta0', 'delta', 'period', 'alpha', 'gamma', 'mu', 'w', 'ar']
    derived_param = [('betaT', 'beta0*(1-delta*cos(2*3.14159*t/period))')]

    # Exposure, internal transmission
    trans_exp=Transition(origin='S', destination='E',transition_type=TransitionType.T)
    event_exp=Event(rate='betaT*S*I/N', transition_list=[trans_exp])

    # Exposure, external attack rate (ar)
    trans_exp_ext=Transition(origin='S', destination='E',transition_type=TransitionType.T)
    event_exp_ext=Event(rate='S*ar', transition_list=[trans_exp_ext])

    # Infectious
    trans_inf=Transition(origin='E', destination='I',transition_type=TransitionType.T)
    event_inf=Event(rate='alpha*E', transition_list=[trans_inf])

    # Recovery
    trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

    # Waning
    trans_wan=Transition(origin='R', destination='S',transition_type=TransitionType.T)
    event_wan=Event(rate='w*R', transition_list=[trans_wan])

    # Transitions in and out of total population count
    # These will be used in the following event definitions
    birth_N=Transition(destination="N", transition_type="B")
    death_N=Transition(origin="N", transition_type="D")

    # 1) Birth event into S
    birth=Transition(destination="S", transition_type="B")
    event_birth=Event(transition_list=[birth, birth_N], rate='mu*N')

    # 2) Death event of an S
    death_S=Transition(origin="S", transition_type="D")
    event_death_S=Event(transition_list=[death_S, death_N], rate='mu*S')

    # 3) Death event of an E
    death_E=Transition(origin="E", transition_type="D")
    event_death_E=Event(transition_list=[death_E, death_N], rate='mu*E')

    # 4) Death event of an I
    death_I=Transition(origin="I", transition_type="D")
    event_death_I=Event(transition_list=[death_I, death_N], rate='mu*I')

    # 5) Death event of an R
    death_R=Transition(origin="R", transition_type="D")
    event_death_R=Event(transition_list=[death_R, death_N], rate='mu*R')

    model=SimulateOde(state=state_list,
                      param=param_list,
                      derived_param=derived_param,
                      event=[event_exp,
                             event_exp_ext,
                             event_inf,
                             event_rec,
                             event_wan,
                             event_birth,
                             event_death_S,
                             event_death_E,
                             event_death_I,
                             event_death_R])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

def SEIR_Multiple(n=2, param=None):
    """
    An SEIR model that describe spatial heterogeneity [Brauer2008]_, page 180.
    The model originated from [Lloyd1996]_ and notations used here
    follows [Brauer2008]_.

    .. math::
        \\frac{dS_{i}}{dt} &= dN_{i} - dS_{i} - \\lambda_{i} S_{i} \\\\
        \\frac{dE_{i}}{dt} &= \\lambda_{i}S_{i} - (d + \\epsilon) E_{i} \\\\
        \\frac{dI_{i}}{dt} &= \\epsilon E_{i} - (d + \\gamma) I_{i} \\\\
        \\frac{dR_{i}}{dt} &= \\gamma I_{i} - dR_{i}

    where

    .. math::
        \\lambda_{i} = \\sum_{j=1}^{n} \\beta_{i,j} I_{j} (1\\{i \\neq j\\} p)

    with :math:`n` being the number of patch and :math:`p` the coupled factor.

    Examples
    --------
    Use the initial conditions that were derived from the stationary condition
    specified in [Brauer2008]_.

    >>> import numpy as np
    >>> from pygom import common_models
    >>> paramEval = {'beta_00':0.0010107, 'beta_01':0.0010107,
    >>>              'beta_10':0.0010107, 'beta_11':0.0010107,
    >>>              'd':0.02,'epsilon':45.6, 'gamma':73.0,
    >>>              'N_0':10**6,'N_1':10**6,'p':0.01}
    >>> x0 = [36139.3224081278, 422.560577637822,
    >>>       263.883351688369, 963174.233662546]
    >>> ode = common_models.SEIR_Multiple()
    >>> t = np.linspace(0, 40, 100)
    >>> x01 = []
    >>> for s in x0:
    >>>     x01 += [s]
    >>>     x01 += [s]
    >>> ode.parameters = paramEval
    >>> ode.initial_values = (x01, t[0])
    >>> solution, output = ode.integrate(t[1::], full_output=True)
    >>> ode.plot()
    """
    if n is None:
        n = 2
    s = [str(i) for i in range(n)]

    beta = []
    lambda_str = []
    lambda_name = []

    state_name = ["S", "E", "I", "R"]
    states = OrderedDict.fromkeys(state_name, [])
    N = []

    for i in s:
        for v in states:
            states[v] = states[v] + [str(v) + "_" + i]
        N += ['N_' + i]
        lambda_temp = '0'
        for j in s:
            beta += ['beta_' + i + j]
            if i==j:
                lambda_temp += '+ I_' + j + '*beta_' + i + j
            else:
                lambda_temp += '+ I_' + j + '*beta_' + i + j + '*p'
        lambda_str += [lambda_temp]
        lambda_name += ['lambda_' + i]

    param_list = beta + ['d', 'epsilon', 'gamma', 'p'] + N

    state_list = []
    for v in states:
        state_list += states[v]

    transition = []
    bd_list = []
    derived_param = []
    for i in range(n):
        derived_param += [(lambda_name[i], lambda_str[i])]
        transition += [Transition(origin=states['S'][i],
                                  destination=states['E'][i],
                                  equation=lambda_name[i] + '*' + states['S'][i],
                                  transition_type=TransitionType.T)]
        transition += [Transition(origin=states['E'][i],
                                  destination=states['I'][i],
                                  equation='epsilon*' + states['E'][i],
                                  transition_type=TransitionType.T)]
        transition += [Transition(origin=states['I'][i],
                                  destination=states['R'][i],
                                  equation='gamma*' + states['I'][i],
                                  transition_type=TransitionType.T)]
        for v in states:
            bd_list += [Transition(origin=states[v][i], equation='d*' + states[v][i], transition_type=TransitionType.D)]
        bd_list += [Transition(origin=states['S'][i], equation='d*' + N[i], transition_type=TransitionType.B)]

    ode_obj = SimulateOde(state_list,
                               param_list,
                               derived_param=derived_param,
                               transition=transition,
                               birth_death=bd_list)
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj

def Influenza_SLIARD(param=None):
    """
    A simple influenza model from [Brauer2008]_, page 323.

    .. math::
        \\frac{dS}{dt} &= -S \\beta (I + \\delta A) \\\\
        \\frac{dL}{dt} &= S \\beta (I + \\delta A) - \\kappa L \\\\
        \\frac{dI}{dt} &= p \\kappa L - \\alpha I \\\\
        \\frac{dA}{dt} &= (1 - p) \\kappa L - \\eta A \\\\
        \\frac{dR}{dt} &= f \\alpha I + \\eta A \\\\
        \\frac{dD}{dt} &= (1 - f) \\alpha I
    """


    state_list = [('S', (0,None)),
                  ('L', (0,None)),
                  ('I', (0, None)),
                  ('A', (0, None)),
                  ('R', (0, None)),
                  ('D', (0, None))]
    
    param_list = ['beta', 'delta', 'N', 'kappa', 'p', 'epsilon', 'alpha', 'f']

    # 1) Susceptibles enter latent phase through:
    ## (i) Encounters with symptomatic infectious, I
    trans_exp_I=Transition(origin='S', destination='L',transition_type=TransitionType.T)
    event_exp_I=Event(rate='beta*S*I/N', transition_list=[trans_exp_I])
    ## (ii) Encounters with asymptomatic infectious, A
    trans_exp_A=Transition(origin='S', destination='L',transition_type=TransitionType.T)
    event_exp_A=Event(rate='beta*S*delta*A/N', transition_list=[trans_exp_A])
    # 2) Latent phase ends
    ## (i) A fraction, p, go on to become symptomatic infectious, I
    trans_inf_I=Transition(origin='L', destination='I',transition_type=TransitionType.T)
    event_inf_I=Event(rate='p*kappa*L', transition_list=[trans_inf_I])
    ## (ii) The remaining (1-p) become asymptomatic infectious, A
    trans_inf_A=Transition(origin='L', destination='A',transition_type=TransitionType.T)
    event_inf_A=Event(rate='(1 - p)*kappa*L', transition_list=[trans_inf_A])
    # 3) All asymptomatics recover
    trans_rec_A=Transition(origin='A', destination='R',transition_type=TransitionType.T)
    event_rec_A=Event(rate='epsilon*A', transition_list=[trans_rec_A])
    # 4) For the symptomatics:
    ## (i) A fraction, f, recover
    trans_rec_I=Transition(origin='I', destination='R',transition_type=TransitionType.T)
    event_rec_I=Event(rate='f*alpha*I', transition_list=[trans_rec_I])
    ## (ii) The remaining 1-f die
    trans_death_I=Transition(origin='I', destination='D',transition_type=TransitionType.T)
    event_death_I=Event(rate='(1-f)*alpha*I', transition_list=[trans_death_I])

    model = SimulateOde(state=state_list,
                        param=param_list,
                        event=[event_exp_I,
                               event_exp_A,
                               event_inf_I,
                               event_inf_A,
                               event_rec_A,
                               event_rec_I,
                               event_death_I])

    if param is None:
        return model
    else:
        model.parameters = param
        return model


def Legrand_Ebola_SEIHFR(param=None):
    """
    The Legrand Ebola model [Legrand2007]_ with 6 compartments that includes the
    H = hospitalization and F = funeral state.
    The set of equations that describes the model are

    .. math::
        \\frac{dS}{dt} &= -(\\beta_{I}SI + \\beta_{H}SH + \\beta_{F}SF) \\\\
        \\frac{dE}{dt} &= (\\beta_{I}SI + \\beta_{H}SH + \\beta_{F}SF) - \\alpha E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\gamma_{H} \\theta_{1} + \\gamma_{I}(1-\\theta_{1})(1-\\delta_{1}) + \\gamma_{D}(1-\\theta_{1})\\delta_{1})I \\\\
        \\frac{dH}{dt} &= \\gamma_{H}\\theta_{1}I - (\\gamma_{DH}\\delta_{2} + \\gamma_{IH}(1-\\delta_{2}))H \\\\
        \\frac{dF}{dt} &= \\gamma_{D}(1-\\theta_{1})\\delta_{1}I + \\gamma_{DH}\\delta_{2}H - \\gamma_{F}F \\\\
        \\frac{dR}{dt} &= \\gamma_{I}(1-\\theta_{1})(1-\\delta_{1})I + \\gamma_{IH}(1-\\delta_{2})H + \\gamma_{F}F.

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> x0 = [200000-3, 3, 0.0, 0.0, 0.0, 0.0]
    >>> t = np.linspace(0, 25, 100)
    >>> ode = common_models.Legrand_Ebola_SEIHFR([('beta_I',0.588),('beta_H',0.794),('beta_F',7.653),('omega_I',10.0/7.0),('omega_D',9.6/7.0),('omega_H',5.0/7.0),('omega_F',2.0/7.0),('alphaInv',7.0/7.0),('delta',0.81),('theta',0.80),('kappa',300.0),('interventionTime',7.0),('N',200000)])
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    state_list = ['S', 'E', 'I', 'H', 'F', 'R']

    param_list = ['beta_I', 'beta_H', 'beta_F',
                  'omega_I', 'omega_D', 'omega_H', 'omega_F',
                  'alphaInv', 'delta', 'theta',
                  'kappa', 'interventionTime', 'N']

    derived_param_list = [
        ('gamma_I', '1/omega_I'),
        ('gamma_D', '1/omega_D'),
        ('gamma_H', '1/omega_H'),
        ('gamma_F', '1/omega_F'),
        ('alpha', '1/alphaInv'),
        ('gamma_IH', '1/((1/gamma_I) - (1/gamma_H))'),
        ('gamma_DH', '1/((1/gamma_D) - (1/gamma_H))'),
        ('delta_1', 'delta*gamma_I/(delta*gamma_I + (1 - delta)*gamma_D)'),
        ('delta_2', 'delta*gamma_IH / (delta*gamma_IH + (1 - delta)*gamma_DH)'),
        ('theta_A', 'theta*(gamma_I*(1 - delta_1) + gamma_D*delta_1)'),
        ('theta_1', 'theta_A/(theta_A + (1 - theta)*gamma_H)'),
        ('t_trans', '(t - interventionTime)/kappa'),
        ('beta_H_Time', 'beta_H*(1 - (0.5*(tanh(t_trans/2)+1)) )'),
        ('beta_F_Time', 'beta_F*(1 - (0.5*(tanh(t_trans/2)+1)) )')
        ]

    transition_list = [
        Transition(origin='S', destination='E',
                   equation='(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F)/N',
                   transition_type=TransitionType.T),
        Transition(origin='E', destination='I',
                   equation='alpha*E',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='H',
                   equation='gamma_H*theta_1*I',
                   transition_type=TransitionType.T),
        Transition(origin='H', destination='F',
                   equation='gamma_DH*delta_2*H',
                   transition_type=TransitionType.T),
        Transition(origin='F', destination='R',
                   equation='gamma_F*F',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R',
                   equation='gamma_I*(1 - theta_1)*(1 - delta_1)*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='F',
                   equation='gamma_D*(1 - theta_1) * delta_1*I',
                   transition_type=TransitionType.T),
        Transition(origin='H', destination='R',
                   equation='gamma_IH*(1 - delta_2)*H',
                   transition_type=TransitionType.T)
        ]

    model = SimulateOde(state=state_list,
                        param=param_list,
                        derived_param=derived_param_list,
                        transition=transition_list)

    # set return, depending on whether we have input the parameters
    if param is None:
        return model
    else:
        model.parameters = param
        return model

def Lotka_Volterra(param=None):
    """
    Standard Lotka-Volterra model with two states and four parameters [Lotka1920]_

    .. math::
        \\frac{dx}{dt} &= \\alpha x - cxy \\\\
        \\frac{dy}{dt} &= -\\delta y + \\gamma xy

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> params = {'alpha':1, 'delta':3, 'c':2, 'gamma':6}
    >>> ode = common_models.Lotka_Volterra(params)
    >>> ode.initial_values = ([2.0, 6.0], 0)
    >>> t = np.linspace(0.1, 100, 10000)
    >>> ode.integrate(t)
    >>> ode.plot()
    """

    state_list = [('x', (0,None)), ('y', (0,None))]         # x=prey, y=pred
    param_list = ['alpha', 'beta', 'gamma', 'delta']

    # 1) Birth of prey
    birth_prey=Transition(destination="x", transition_type="B")
    event_birth_prey=Event(transition_list=[birth_prey], rate='alpha*x')

    # 2) Death of prey
    death_prey=Transition(origin="x", transition_type="D")
    event_death_prey=Event(transition_list=[death_prey], rate='beta*x*y')

    # 3) Birth of predator
    birth_pred=Transition(destination="y", transition_type="B")
    event_birth_pred=Event(transition_list=[birth_pred], rate='delta*x*y')

    # 4) Death of predator
    death_pred=Transition(origin="y", transition_type="D")
    event_death_pred=Event(transition_list=[death_pred], rate='gamma*y')

    model=SimulateOde(state=state_list,
                      param=param_list,
                      event=[event_birth_prey,
                             event_death_prey,
                             event_birth_pred,
                             event_death_pred])

    if param is None:
        return model
    else:
        model.parameters = param
        return model

# Doesn't seem like a very useful formulation. 

# def Lotka_Volterra_4State(param=None):
#     """
#     The four state Lotka-Volterra model [Lotka1920]_. A common interpretation is that
#     a = Grass, x = rabbits, y = foxes and b is the death of foxes.

#     .. math::
#         \\frac{da}{dt} &= k_{0} a x \\\\
#         \\frac{dx}{dt} &= k_{0} a x - k_{1} x y \\\\
#         \\frac{dy}{dt} &= k_{1} x y - k_{2} y \\\\
#         \\frac{db}{dt} &= k_{2} y

#     Examples
#     --------
#     >>> import numpy as np
#     >>> from pygom import common_models
#     >>> x0 = [150.0, 10.0, 10.0, 0.0]
#     >>> t = np.linspace(0, 15, 100)
#     >>> params = [0.01, 0.1, 1.0]
#     >>> ode = common_models.Lotka_Volterra_4State(params)
#     >>> ode.initial_values = (x0, t[0])
#     >>> ode.integrate(t[1::])
#     >>> ode.plot()
#     """

#     state_list = ['a', 'x', 'y']
#     param_list = ['k0', 'k1', 'k2']

#     # 1) Grass turning into rabbits
#     trans_grazing=Transition(origin='a', destination='x',transition_type=TransitionType.T)
#     event_grazing=Event(rate='k0*a*x', transition_list=[trans_grazing])

#     # 2) Foxes eating rabbits
#     trans_hunting=Transition(origin='x', destination='y',transition_type=TransitionType.T)
#     event_hunting=Event(rate='k1*x*y', transition_list=[trans_hunting])

#     # 3) Foxes natural death rate
#     death_pred=Transition(origin="y", transition_type="D")
#     event_death_pred=Event(transition_list=[death_pred], rate='k2*y')

#     model=SimulateOde(state=state_list,
#                       param=param_list,
#                       event=[event_birth_prey,
#                              event_death_prey,
#                              event_birth_pred,
#                              event_death_pred])

#     if param is None:
#         return model
#     else:
#         model.parameters = param
#         return model


def Robertson(param=None):
    """
    The so called Robertson problem [Robertson1966]_, which is a standard example used to
    test stiff integrator.

    .. math::
        \\frac{dy_{1}}{dt} &= -0.04 y_{1} + 1 \\cdot 10^{4} y_{2} y_{3} \\\\
        \\frac{dy_{2}}{dt} &= 0.04 y_{1} - 1 \\cdot 10^{4} y_{2} y_{3} - 3 \\cdot 10^{7} y_{2}^{2}\\\\
        \\frac{dy_{3}}{dt} &= 3 \\cdot 10^{7} y_{2}^{2}

    Examples
    --------

    >>> from pygom import common_models
    >>> import numpy
    >>> t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
    >>> ode = common_models.Robertson()
    >>> ode.initial_values = ([1.0,0.0,0.0], t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot() # note that this is not being plotted in the log scale
    """
    # note how we have short handed the definition
    state = ['y1:4']
    # note that we do not have any parameters, or rather,
    # we have hard coded in the parameters
    param_list = []

    trans_1_2=Transition(origin='y1', destination='y2',transition_type=TransitionType.T)
    event_1_2=Event(rate='0.04*y1', transition_list=[trans_1_2])

    trans_2_1=Transition(origin='y2', destination='y1',transition_type=TransitionType.T)
    event_2_1=Event(rate='1e4*y2*y3', transition_list=[trans_2_1])

    trans_2_3=Transition(origin='y2', destination='y3',transition_type=TransitionType.T)
    event_2_3=Event(rate='3e7*y2*y2', transition_list=[trans_2_3])

    # initialize the model
    model = SimulateOde(state, param_list, event=[event_1_2,
                                                  event_2_1,
                                                  event_2_3])

    if param is None:
        return model
    else:
        raise Warning("Input parameters not used")

#############################################
# ODEs
#############################################


def SIR_norm(param=None):
    """
    A normalized SIR model:

    .. math::
        \\frac{dS}{dt} &= -\\beta SI \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I


    Examples
    --------
    The model that produced top two graph in Figure 1.3 of the reference above.
    First, when everyone is susceptible and only one individual was infected.

    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SIR({'beta':3.6, 'gamma':0.2})
    >>> t = np.linspace(0, 730, 1001)
    >>> N = 7781984.0
    >>> x0 = [1.0, 10/N, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    Second model with a more *realistic* scenario

    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SIR({'beta':3.6, 'gamma':0.2})
    >>> t = np.linspace(0, 730, 1001)
    >>> N = 7781984.0
    >>> x0 = [0.065, 123*(5.0/30.0)/N, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    """
    state = ['S', 'I', 'R']
    param_list = ['beta', 'gamma']

    dSdt=Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.ODE)
    dIdt=Transition(origin='I', equation='beta*S*I-gamma*I', transition_type=TransitionType.ODE)
    dRdt=Transition(origin='R', equation='gamma*I', transition_type=TransitionType.ODE)

    # initialize the model
    ode_obj = SimulateOde(state=state, param=param_list, ode=[dSdt, dIdt, dRdt])

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def FitzHugh(param=None):
    """
    The standard FitzHugh model without external input [FitzHugh1961]_

    .. math::
        \\frac{dV}{dt} &=  c ( V - \\frac{V^{3}}{3} + R) \\\\
        \\frac{dR}{dt} &= -\\frac{1}{c}(V - a + bR).

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.FitzHugh({'a':0.2, 'b':0.2, 'c':3.0})
    >>> t = np.linspace(0, 20, 101)
    >>> x0 = [1.0, -1.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    # the two states
    state = ['V', 'R']
    # and the three parameters
    param_list = ['a', 'b', 'c']

    # the set of ode
    ode = [
        Transition(origin='V', equation='c*(V - (V*V*V)/3 + R)',
                   transition_type=TransitionType.ODE),
        Transition(origin='R', equation='-( (V - a + b*R)/c )',
                   transition_type=TransitionType.ODE)
        ]
    # setup our ode
    ode_obj = SimulateOde(state, param_list,
                               derived_param=None,
                               transition=None,
                               birth_death=None,
                               ode=ode)
    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def Lorenz(param=None):
    """
    Lorenz attractor define by three parameters, :math:`\\beta,\\sigma,\\rho`
    as per [Lorenz1963]_.

    .. math::
        \\frac{dx}{dt} &= \\sigma (y-x) \\\\
        \\frac{dy}{dt} &= x (\\rho - z) - y \\\\
        \\frac{dz}{dt} &= xy - \\beta z

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy
    >>> from pygom import common_models
    >>> t = numpy.linspace(0, 20, 101)
    >>> params = {'beta':8.0/3.0, 'sigma':10.0, 'rho':28.0}
    >>> ode = common_models.Lorenz(params)
    >>> ode.initial_values = ([1., 1., 1.], t[0])
    >>> solution = ode.integrate(t[1::])
    >>> plt.plot(solution[:,0], solution[:,2])
    >>> plt.show()
    """

    state = ['x', 'y', 'z']
    param_list = ['beta', 'sigma', 'rho']
    ode = [
        Transition(origin='x', equation='sigma*(y - x)',
                   transition_type=TransitionType.ODE),
        Transition(origin='y', equation='x*(rho - z) - y',
                   transition_type=TransitionType.ODE),
        Transition(origin='z', equation='x*y - beta*z',
                   transition_type=TransitionType.ODE)
        ]
    # initialize the model
    ode_obj = SimulateOde(state, param_list, ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def vanDerPol(param=None):
    """
    The van der Pol equation [vanderpol1926]_, a second order ode

    .. math::
        y^{\\prime\\prime} - \\mu (1-y^{2}) y^{\\prime} + y = 0

    where :math:`\\mu > 0`.  This can be converted to a first
    order ode by equating :math:`x = y^{\\prime}`

    .. math::
        x^{\\prime} - \\mu (1 - y^{2}) x + y = 0

    which result in a coupled ode

    .. math::
        x^{\\prime} &= \\mu (1 - y^{2}) x - y \\\\
        y^{\\prime} &= x

    and this can be solved via standard method.

    Examples
    --------
    >>> from pygom import common_models
    >>> import numpy
    >>> t = numpy.linspace(0, 20, 1000)
    >>> ode = common_models.vanDerPol({'mu':1.0})
    >>> ode.initial_values = ([2.0,0.0], t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    state_list = ['y', 'x']
    param_list = ['mu']
    ode = [
        Transition(origin='y', equation='x',
                   transition_type=TransitionType.ODE),
        Transition(origin='x', equation='mu*(1 - y*y)*x -  y',
                   transition_type=TransitionType.ODE)
        ]
    # initialize the model
    ode_obj = SimulateOde(state_list, param_list, ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj
