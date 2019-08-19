"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    A set of commonly used models

"""

from collections import OrderedDict

from .transition import TransitionType, Transition
from .deterministic import DeterministicOde


def SIS(param=None):
    """
    A standard SIS model

    .. math::
        \\frac{dS}{dt} &= -\\beta SI + \\gamma I \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SIS({'beta':0.5, 'gamma':0.2})
    >>> t = np.linspace(0, 20, 101)
    >>> x0 = [1.0, 0.1]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    state = ['S', 'I']
    param_list = ['beta', 'gamma']
    transition = [
        Transition(origin='S', destination='I', equation='beta*S*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='S', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]
    # initialize the model
    ode = DeterministicOde(state,
                           param_list,
                           transition=transition)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.parameters = param
        return ode


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

    state = ['I', 'tau']
    param_list = ['alpha']
    derived_param = [('betaT', '2 - 1.8*cos(5*tau)')]
    ode = [
        Transition(origin='I',
                   equation='(betaT - alpha)*I - betaT*I*I',
                   transition_type=TransitionType.ODE),
        Transition(origin='tau',
                   equation='1',
                   transition_type=TransitionType.ODE)
        ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list,
                               derived_param=derived_param,
                               ode=ode)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def SIR(param=None):
    """
    A standard SIR model as per [Brauer2008]_

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
    transition = [
        Transition(origin='S', destination='I', equation='beta*S*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list, transition=transition)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def SIR_N(param=None):
    """
    A standard SIR model [Brauer2008]_ with population N.  This is the unnormalized
    version of the SIR model.

    .. math::
        \\frac{dS}{dt} &= -\\beta SI/N \\\\
        \\frac{dI}{dt} &= \\beta SI/N- \\gamma I \\\\
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
    >>> x0 = [N, 1.0, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    Second model with a more *realistic* scenario

    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SIR({'beta':3.6, 'gamma':0.2})
    >>> t = np.linspace(0, 730, 1001)
    >>> N = 7781984.0
    >>> x0 = [int(0.065*N), 21.0, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """
    state = ['S', 'I', 'R']
    param_list = ['beta', 'gamma', 'N']
    transition = [
        Transition(origin='S', destination='I', equation='beta*S*I/N',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list, transition=transition)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


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
    state = ['S', 'I', 'R']
    param_list = ['beta', 'gamma', 'B', 'mu']
    transition = [
        Transition(origin='S', destination='I', equation='beta*S*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]
    # our birth and deaths
    birth_death = [
        Transition(origin='S', equation='B',
                   transition_type=TransitionType.B),
        Transition(origin='S', equation='mu*S',
                   transition_type=TransitionType.D),
        Transition(origin='I', equation='mu*I',
                   transition_type=TransitionType.D)
        ]

    # initialize the model
    ode_obj = DeterministicOde(state, param_list,
                               birth_death=birth_death,
                               transition=transition)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def SEIR(param=None):
    """
    A standard SEIR model [Brauer2008]_, defined by the ode

    .. math::
        \\frac{dS}{dt} &= -\\beta SI \\\\
        \\frac{dE}{dt} &= \\beta SI - \\alpha E \\\\
        \\frac{dI}{dt} &= \\alpha E - \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I

    Examples
    --------

    >>> import numpy as np
    >>> from pygom import common_models
    >>> ode = common_models.SEIR({'beta':1800, 'gamma':100, 'alpha':35.84})
    >>> t = np.linspace(0, 50, 1001)
    >>> x0 = [0.0658, 0.0007, 0.0002, 0.0]
    >>> ode.initial_values = (x0, t[0])
    >>> solution,output = ode.integrate(t[1::], full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`.SEIR_Birth_Death`
    """

    state = ['S', 'E', 'I', 'R']
    param_list = ['beta', 'alpha', 'gamma']

    transition = [
        Transition(origin='S', destination='E', equation='beta*S*I',
                   transition_type=TransitionType.T),
        Transition(origin='E', destination='I', equation='alpha*E',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]

    ode_obj = DeterministicOde(state, param_list, transition=transition)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


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

    state = ['S', 'E', 'I', 'R']
    param_list = ['beta', 'alpha', 'gamma', 'mu']

    transition = [
        Transition(origin='S', destination='E', equation='beta*S*I',
                   transition_type=TransitionType.T),
        Transition(origin='E', destination='I', equation='alpha*E',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R', equation='gamma*I',
                   transition_type=TransitionType.T)
        ]

    bd_list = [
        Transition(origin='S', equation='mu*S',
                   transition_type=TransitionType.D),
        Transition(origin='E', equation='mu*E',
                   transition_type=TransitionType.D),
        Transition(origin='I', equation='mu*I',
                   transition_type=TransitionType.D),
        Transition(origin='S', equation='mu',
                   transition_type=TransitionType.B)
        ]

    ode_obj = DeterministicOde(state, param_list,
                               transition=transition,
                               birth_death=bd_list)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj

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
    state = ['S', 'E', 'I', 'tau']
    param_list = ['mu', 'alpha', 'gamma', 'beta_0', 'beta_1']
    derived_param = [('beta_S', 'beta_0 * (1 + beta_1*cos(2*pi*tau))')]
    ode = [
        Transition(origin='S', equation='mu - beta_S*S*I - mu*S',
                   transition_type=TransitionType.ODE),
        Transition(origin='E', equation='beta_S*S*I - (mu + alpha)*E',
                   transition_type=TransitionType.ODE),
        Transition(origin='I', equation='alpha*E - (mu + gamma)*I',
                   transition_type=TransitionType.ODE),
        Transition(origin='tau', equation='1',
                   transition_type=TransitionType.ODE)
        ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list,
                               derived_param=derived_param,
                               ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj

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

    ode_obj = DeterministicOde(state_list,
                               param_list,
                               derived_param=derived_param,
                               transition=transition,
                               birth_death=bd_list)
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj

def Influenza_SLIARN(param=None):
    """
    A simple influenza model from [Brauer2008]_, page 323.

    .. math::
        \\frac{dS}{dt} &= -S \\beta (I + \\delta A) \\\\
        \\frac{dL}{dt} &= S \\beta (I + \\delta A) - \\kappa L \\\\
        \\frac{dI}{dt} &= p \\kappa L - \\alpha I \\\\
        \\frac{dA}{dt} &= (1 - p) \\kappa L - \\eta A \\\\
        \\frac{dR}{dt} &= f \\alpha I + \\eta A \\\\
        \\frac{dN}{dt} &= -(1 - f) \\alpha I
    """

    state = ['S', 'L', 'I', 'A', 'R', 'N']
    param_list = ['beta', 'p', 'kappa', 'alpha', 'f', 'delta', 'epsilon']
    ode = [
        Transition(origin='S', equation='-beta*S*(I + delta*A)',
                   transition_type=TransitionType.ODE),
        Transition(origin='L', equation='beta*S*(I + delta*A) - kappa*L',
                   transition_type=TransitionType.ODE),
        Transition(origin='I', equation='p*kappa*L - alpha*I',
                   transition_type=TransitionType.ODE),
        Transition(origin='A', equation='(1 - p)*kappa*L - epsilon*A',
                   transition_type=TransitionType.ODE),
        Transition(origin='R', equation='f*alpha*I + epsilon*A',
                   transition_type=TransitionType.ODE),
        Transition(origin='N', equation='-(1 - f)*alpha*I',
                   transition_type=TransitionType.ODE)
    ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list, ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def Legrand_Ebola_SEIHFR(param=None):
    """
    The Legrand Ebola model [Legrand2007]_ with 6 compartments that includes the
    H = hospitalization and F = funeral state. Note that because this
    is an non-autonomous system, there are in fact a total of 7 states
    after conversion.  The set of equations that describes the model are

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
    >>> x0 = [1.0, 3.0/200000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> t = np.linspace(0, 25, 100)
    >>> ode = common_models.Legrand_Ebola_SEIHFR([('beta_I',0.588),('beta_H',0.794),('beta_F',7.653),('omega_I',10.0/7.0),('omega_D',9.6/7.0),('omega_H',5.0/7.0),('omega_F',2.0/7.0),('alphaInv',7.0/7.0),('delta',0.81),('theta',0.80),('kappa',300.0),('interventionTime',7.0)])
    >>> ode.initial_values = (x0, t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    """

    # define our states
    state = ['S', 'E', 'I', 'H', 'F', 'R', 'tau']
    # and initial parameters
    params = ['beta_I', 'beta_H', 'beta_F',
              'omega_I', 'omega_D', 'omega_H', 'omega_F',
              'alphaInv', 'delta', 'theta',
              'kappa', 'interventionTime']

    # we now construct a list of the derived parameters
    # which has 2 item
    # name
    # equation
    derived_param = [
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
        ('beta_H_Time', 'beta_H*(1 - (1/ (1 + exp(-kappa*(tau - interventionTime)))))'),
        ('beta_F_Time', 'beta_F*(1 - (1/ (1 + exp(-kappa*(tau - interventionTime)))))')
        ]

    # alternatively, we can do it on the operate ode model
    ode_obj = DeterministicOde(state, params)
    # add the derived parameter
    ode_obj.derived_param_list = derived_param

    # define the set of transitions
    # name of origin state
    # name of target state
    # equation
    # type of equation, which is a transition between two state in this case

    transition = [
        Transition(origin='S', destination='E',
                   equation='(beta_I*S*I + beta_H_Time*S*H + beta_F_Time*S*F)',
                   transition_type=TransitionType.T),
        Transition(origin='E', destination='I',
                   equation='alpha*E',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='H',
                   equation='gamma_H*theta_1*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='F',
                   equation='gamma_D*(1 - theta_1) * delta_1*I',
                   transition_type=TransitionType.T),
        Transition(origin='I', destination='R',
                   equation='gamma_I*(1 - theta_1)*(1 - delta_1)*I',
                   transition_type=TransitionType.T),
        Transition(origin='H', destination='F',
                   equation='gamma_DH*delta_2*H',
                   transition_type=TransitionType.T),
        Transition(origin='H', destination='R',
                   equation='gamma_IH*(1 - delta_2)*H',
                   transition_type=TransitionType.T),
        Transition(origin='F', destination='R',
                   equation='gamma_F*F',
                   transition_type=TransitionType.T)
        ]

    bd_list = [Transition(origin='tau', equation='1',
                          transition_type=TransitionType.B)]

    # see how we can insert the transitions later, after initializing the ode object
    # this is not the preferred choice though
    ode_obj.transition_list = transition
    ode_obj.birth_death_list = bd_list
    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


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

    # our two state and four parameters
    # no idea why they are not in capital
    state = ['x', 'y']
    # while these 4 are
    param_list = ['alpha', 'delta', 'c', 'gamma']
    # then define the set of ode
    ode = [
        Transition(origin='x', equation='alpha*x - c*x*y',
                   transition_type=TransitionType.ODE),
        Transition(origin='y', equation='-delta*y + gamma*x*y',
                   transition_type=TransitionType.ODE)
        ]

    ode_obj = DeterministicOde(state, param_list, ode=ode)
    # set return, depending on whether we have input the parameters
    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def Lotka_Volterra_4State(param=None):
    """
    The four state Lotka-Volterra model [Lotka1920]_. A common interpretation is that
    a = Grass, x = rabbits, y = foxes and b is the death of foxes.

    .. math::
        \\frac{da}{dt} &= k_{0} a x \\\\
        \\frac{dx}{dt} &= k_{0} a x - k_{1} x y \\\\
        \\frac{dy}{dt} &= k_{1} x y - k_{2} y \\\\
        \\frac{db}{dt} &= k_{2} y

    Examples
    --------
    >>> import numpy as np
    >>> from pygom import common_models
    >>> x0 = [150.0, 10.0, 10.0, 0.0]
    >>> t = np.linspace(0, 15, 100)
    >>> params = [0.01, 0.1, 1.0]
    >>> ode = common_models.Lotka_Volterra_4State(params)
    >>> ode.initial_values = (x0, t[0])
    >>> ode.integrate(t[1::])
    >>> ode.plot()
    """

    # four states
    state = ['a', 'x', 'y', 'b']
    # three parameters
    param_list = ['k0', 'k1', 'k2']

    # then define the set of ode
    transition = [
        Transition(origin='a', destination='x',
                   equation='k0*a*x',
                   transition_type=TransitionType.T),
        Transition(origin='x', destination='y',
                   equation='k1*x*y',
                   transition_type=TransitionType.T),
        Transition(origin='y', destination='b',
                   equation='k2*y',
                   transition_type=TransitionType.T)
        ]

    ode_obj = DeterministicOde(state, param_list, transition=transition)

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
    ode_obj = DeterministicOde(state, param_list,
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
    ode_obj = DeterministicOde(state, param_list, ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


def vanDelPol(param=None):
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
    >>> ode = common_models.vanDelPol({'mu':1.0})
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
    ode_obj = DeterministicOde(state_list, param_list, ode=ode)

    if param is None:
        return ode_obj
    else:
        ode_obj.parameters = param
        return ode_obj


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
    transition = [
        Transition(origin='y1', destination='y2',
                   equation='0.04*y1',
                   transition_type=TransitionType.T),
        Transition(origin='y2', destination='y1',
                   equation='1e4*y2*y3',
                   transition_type=TransitionType.T),
        Transition(origin='y2', destination='y3',
                   equation='3e7*y2*y2',
                   transition_type=TransitionType.T)
        ]
    # initialize the model
    ode_obj = DeterministicOde(state, param_list, transition=transition)

    if param is None:
        return ode_obj
    else:
        raise Warning("Input parameters not used")
