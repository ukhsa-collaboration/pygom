"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    A set of commonly used models

"""

from pygom.model.transition import TransitionType, Transition
from pygom.model.deterministic import OperateOdeModel
from collections import OrderedDict

def SIS(param=None):
    '''
    A standard SIS model

    .. math::
        \\frac{dS}{dt} &= -\\beta SI + \\gamma I \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I

    Examples
    --------

    >>> ode = common_models.SIS({'beta':0.5,'gamma':0.2})
    >>> t = numpy.linspace(0,20,101)
    >>> x0 = [1.0,0.1]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''

    stateList = ['S', 'I']
    paramList = ['beta', 'gamma']
    transitionList = [
        Transition(origState='S', destState='I', equation='beta * S * I',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='S', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          transitionList=transitionList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        return ode.setParameters(param)


def SIS_Periodic(param=None):
    '''
    A SIS model with periodic contact, defined by the ode

    .. math::
        \\frac{dI}{dt} = (\\beta(t)N - \\alpha) I - \\beta(t)I^{2}

    where

    .. math::
        \\beta(t) = 2 - 1.8 \\cos(5t).

    As the name suggests, it achieves a (stable) periodic solution.

    References
    ----------
    .. [1] Asymptotic behavior in a deterministic epidemic model,
           Hethcote Herbert W, Bulletin of Mathematical Biology,
           Volume 35, pg. 607-614, 1973

    Examples
    --------

    >>> ode = common_models.SIS_Periodic({'alpha':1.0})
    >>> t = numpy.linspace(0,10,101)
    >>> x0 = [0.1,0.]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''
    stateList = ['I', 'tau']
    paramList = ['alpha']
    derivedParamList = [('betaT', '2 - 1.8 * cos(5*tau)')]
    odeList = [
        Transition(origState='I',
                   equation='(betaT - alpha)* I - betaT * I * I',
                   transitionType=TransitionType.ODE),
        Transition(origState='tau',
                   equation='1',
                   transitionType=TransitionType.ODE)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          derivedParamList=derivedParamList,
                          odeList=odeList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        return ode.setParameters(param)

def SIR(param=None):
    '''
    A standard SIR model

    .. math::
        \\frac{dS}{dt} &= -\\beta SI \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I

    References
    ----------
    .. [1] Mathematical Epidemiology, Lecture Notes in Mathematics,
           Brauer Fred, Springer 2008

    Examples
    --------
    The model that produced top two graph in Figure 1.3 of the reference above.
    First, when everyone is susceptible and only one individual was infected.

    >>> ode = common_models.SIR({'beta':3.6,'gamma':0.2})
    >>> t = numpy.linspace(0,730,1001)
    >>> N = 7781984.0
    >>> x0 = [1.0,10/N,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    Second model with a more *realistic* scenario

    >>> ode = common_models.SIR({'beta':3.6,'gamma':0.2})
    >>> t = numpy.linspace(0,730,1001)
    >>> N = 7781984.0
    >>> x0 = [0.065,123*(5.0/30.0)/N,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''
    stateList = ['S', 'I', 'R']
    paramList = ['beta', 'gamma']
    transitionList = [
        Transition(origState='S', destState='I', equation='beta * S * I',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          transitionList=transitionList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def SIR_N(param=None):
    '''
    A standard SIR model with population N.  This is the unnormalized
    version of the SIR model.

    .. math::
        \\frac{dS}{dt} &= -\\beta SI/N \\\\
        \\frac{dI}{dt} &= \\beta SI /N- \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I

    References
    ----------
    .. [1] Mathematical Epidemiology, Lecture Notes in Mathematics,
           Brauer Fred, Springer 2008

    Examples
    --------
    The model that produced top two graph in Figure 1.3 of the reference above.
    First, when everyone is susceptible and only one individual was infected.

    >>> ode = common_models.SIR({'beta':3.6,'gamma':0.2})
    >>> t = numpy.linspace(0,730,1001)
    >>> N = 7781984.0
    >>> x0 = [1.0,10/N,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    Second model with a more *realistic* scenario

    >>> ode = common_models.SIR({'beta':3.6,'gamma':0.2})
    >>> t = numpy.linspace(0,730,1001)
    >>> N = 7781984.0
    >>> x0 = [0.065,123*(5.0/30.0)/N,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''
    stateList = ['S', 'I', 'R']
    paramList = ['beta', 'gamma','N']
    transitionList = [
        Transition(origState='S', destState='I', equation='beta * S * I / N',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          transitionList=transitionList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode


def SIR_Birth_Death(param=None):
    '''
    Extension of the standard SIR model to also include birth and death

    .. math::
        \\frac{dS}{dt} &= B -\\beta SI - \\mu S \\\\
        \\frac{dI}{dt} &= \\beta SI - \\gamma I - \\mu I \\\\
        \\frac{dR}{dt} &= \\gamma I

    References
    ----------
    .. [1] Mathematical Epidemiology, Lecture Notes in Mathematics,
           Brauer Fred, Springer 2008

    Examples
    --------
    The model that produced bottom graph in Figure 1.3 of the reference above.

    >>> B = 126372.0/365.0
    >>> N = 7781984.0
    >>> ode = common_models.SIR_Birth_Death({'beta':3.6,'gamma':0.2,'B':B/N,'mu':B/N})
    >>> t = numpy.linspace(0,35*365,10001)
    >>> x0 = [0.065,123.0*(5.0/30.0)/N,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution,output = ode.integrate(t[1::],full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`SIR`
    '''
    stateList = ['S', 'I', 'R']
    paramList = ['beta', 'gamma', 'B', 'mu']
    transitionList = [
        Transition(origState='S', destState='I', equation='beta * S * I',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]
    # our birth and deaths
    birthDeathList = [
        Transition(origState='S', equation='B', transitionType=TransitionType.B),
        Transition(origState='S', equation='mu * S', transitionType=TransitionType.D),
        Transition(origState='I', equation='mu * I', transitionType=TransitionType.D)
        ]

    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          birthDeathList=birthDeathList,
                          transitionList=transitionList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def SEIR(param=None):
    '''
    A standard SEIR model, defined by the ode

    .. math::
        \\frac{dS}{dt} &= -\\beta SI \\\\
        \\frac{dE}{dt} &= \\beta SI - \\alpha E \\\\
        \\frac{dI}{dt} &= \\alpha E - \\gamma I \\\\
        \\frac{dR}{dt} &= \\gamma I

    Examples
    --------

    >>> ode = common_models.SEIR({'beta':1800,'gamma':100,'alpha':35.84})
    >>> t = numpy.linspace(0,50,1001)
    >>> x0 = [0.0658,0.0007,0.0002,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution,output = ode.integrate(t[1::],full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`SEIR_Birth_Death`
    '''

    stateList = ['S', 'E', 'I', 'R']
    paramList = ['beta', 'alpha', 'gamma']

    transitionList = [
        Transition(origState='S', destState='E', equation='beta * S * I',
                   transitionType=TransitionType.T),
        Transition(origState='E', destState='I', equation='alpha * E',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]

    ode = OperateOdeModel(stateList,
                          paramList,
                          transitionList=transitionList)

    if param is None:
        return ode
    else:
        return ode.setParameters(param)

def SEIR_Birth_Death(param=None):
    '''
    A standard SEIR model with birth and death, defined by the ode

    .. math::
        \\frac{dS}{dt} &= \\mu - \\beta SI - \\mu S \\\\
        \\frac{dE}{dt} &= \\beta SI - (\\mu + \\alpha) E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\mu + \\gamma) I \\\\
        \\frac{dR}{dt} &= \\gamma I

    References
    ----------
    .. [1] Seasonality and period-doubling bifurcations in an epidemic model,
           Aron J.L. and Schwartz I.B., Journal of Theoretical Biology,
           Volume 110, Issue 4, pg 665-679, 1984

    Examples
    --------
    Uses the same set of parameters as the examples in :func:`.SEIR`
    apart from :math:`\mu` which is new.

    >>> params = {'beta':1800,'gamma':100,'alpha':35.84,'mu':0.02}
    >>> ode = common_models.SEIR_Birth_Death(params)
    >>> t = numpy.linspace(0,50,1001)
    >>> x0 = [0.0658,0.0007,0.0002,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution,output = ode.integrate(t[1::],full_output=True)
    >>> ode.plot()

    See also
    --------
    :func:`SEIR`
    '''

    stateList = ['S', 'E', 'I', 'R']
    paramList = ['beta', 'alpha', 'gamma', 'mu']

    transitionList = [
        Transition(origState='S', destState='E', equation='beta * S * I',
                   transitionType=TransitionType.T),
        Transition(origState='E', destState='I', equation='alpha * E',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R', equation='gamma * I',
                   transitionType=TransitionType.T)
        ]

    bdList = [
        Transition(origState='S', equation='mu * S', transitionType=TransitionType.D),
        Transition(origState='E', equation='mu * E', transitionType=TransitionType.D),
        Transition(origState='I', equation='mu * I', transitionType=TransitionType.D),
        Transition(origState='S', equation='mu', transitionType=TransitionType.B)
        ]

    ode = OperateOdeModel(stateList,
                          paramList,
                          transitionList=transitionList,
                          birthDeathList=bdList)

    if param is None:
        return ode
    else:
        return ode.setParameters(param)

def SEIR_Birth_Death_Periodic(param=None):
    '''
    A SEIR birth death model with periodic contact, defined by the ode

    .. math::
        \\frac{dS}{dt} &= \\mu - \\beta(t)SI - \\mu S \\\\
        \\frac{dE}{dt} &= \\beta(t)SI - (\\mu + \\alpha) E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\mu + \\gamma) I \\\\
        \\frac{dR}{dt} &= \\gamma I

    where

    .. math::
        \\beta(t) = \\beta_{0} (1 + \\beta_{1} \\cos(2 \\pi t)).

    An extension of an SEIR birth death model by varying the contact rate through time.

    References
    ----------
    .. [1] Seasonality and period-doubling bifurcations in an epidemic model,
           Aron J.L. and Schwartz I.B., Journal of Theoretical Biology,
           Volume 110, Issue 4, pg 665-679, 1984

    Examples
    --------
    Uses the same set of parameters as the examples in :func:`SEIR_Birth_Death` but
    now we have two beta parameters instead of one.

    >>> params = {'beta0':1800,'beta1':0.2,'gamma':100,'alpha':35.84,'mu':0.02}
    >>> ode = common_models.SEIR_Birth_Death_Periodic(params)
    >>> t = numpy.linspace(0,50,1001)
    >>> x0 = [0.0658,0.0007,0.0002,0.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution,output = ode.integrate(t[1::],full_output=True)
    >>> ode.plot()
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(numpy.log(solution[:,0]),numpy.log(solution[:,1]))
    >>> plt.show()
    >>> plt.plot(numpy.log(solution[:,0]),numpy.log(solution[:,2]))
    >>> plt.show()

    See also
    --------
    :func:`SEIR`,:func:`SEIR_Birth_Death`,:func:`SIR_Periodic`

    '''
    stateList = ['S', 'E', 'I', 'tau']
    paramList = ['mu', 'alpha', 'gamma', 'beta_0', 'beta_1']
    derivedParamList = [('beta_S', 'beta_0 * (1 + beta_1 * cos(2 * pi * tau))')]
    odeList = [
        Transition(origState='S', equation='mu - beta_S * S * I - mu * S',
                   transitionType=TransitionType.ODE),
        Transition(origState='E', equation='beta_S * S * I - (mu + alpha) * E',
                   transitionType=TransitionType.ODE),
        Transition(origState='I', equation='alpha * E - (mu + gamma) * I',
                   transitionType=TransitionType.ODE),
        Transition(origState='tau', equation='1',
                   transitionType=TransitionType.ODE)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          derivedParamList=derivedParamList,
                          odeList=odeList)

    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode
    
def SEIR_Multiple(n=2, param=None):
    '''
    An SEIR model that describe spatial heterogeneity [1], page 180.  The
    model originated from [2] and notations used here follows [1].

    .. math::
        \\frac{dS_{i}}{dt} &= dN_{i} - dS_{i} - \\lambda_{i}S_{i} \\\\
        \\frac{dE_{i}}{dt} &= \\lambda_{i}S_{i} - (d+\\epsilon)E_{i} \\\\
        \\frac{dI_{i}}{dt} &= \\epsilon E_{i} - (d+\\gamma) I_{i} \\\\
        \\frac{dR_{i}}{dt} &= \\gamma I_{i} - dR_{i}
    
    where 
    
    .. math::
        \\lambda_{i} = \\sum_{j=1}^{n} \\beta_{i,j} I_{j} (1\\{i\neqj\\} p)
        
    with :math:`n` being the number of patch and :math:`p` the coupled factor.

    Examples
    --------
    Use the initial conditions that were derived from the stationary condition
    specified in [2].

    >>> paramEval = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,
    >>>              'beta_11':0.0010107,'d':0.02,'epsilon':45.6,'gamma':73.0,
    >>>              'N_0':10**6,'N_1':10**6,'p':0.01}
    >>> x0 = [36139.3224081278, 422.560577637822, 263.883351688369, 963174.233662546]
    >>> ode = common_models.SEIR_Multiple()
    >>> t = numpy.linspace(0,40,100)
    >>> x01 = []
    >>> for s in x0:
    >>>     x01 += [s]
    >>>     x01 += [s]
    >>> ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
    >>> solution,output = ode.integrate(t[1::],full_output=True)
    >>> ode.plot()

    References
    ----------
    .. [1] Mathematical Epidemiology, Lecture Notes in Mathematics,
           Brauer Fred, Springer 2008
    .. [2] Lloyd A.L. and May R.M., Spatial Heterogeneity in Epidemic Models,
           Journal of Theoretical Biology, Vol 179, no. 1, pg 1-11, 1996
    '''
    if n is None:
        n = 2
    s = [str(i) for i in range(n)]
        
    beta = []
    lambdaStr = []
    lambdaName = []

    stateName = ["S","E","I","R"]
    states = OrderedDict.fromkeys(stateName,[])
    N =  []

    for i in s:
        for v in states:
            states[v] = states[v]+[str(v)+"_"+i]
        N += ['N_'+i]
        lambdaTemp = '0'
        for j in s: 
            beta += ['beta_'+i+j]
            if i==j:
                lambdaTemp += '+ I_'+j+'*beta_'+i+j
            else:
                lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
        lambdaStr += [lambdaTemp]
        lambdaName += ['lambda_'+i]

    paramList = beta + ['d','epsilon','gamma','p'] + N

    stateList = []
    for v in states: stateList += states[v]

    transitionList = []
    bdList = []
    derivedParamList = []
    for i in range(n):
        derivedParamList += [(lambdaName[i],lambdaStr[i])]
        transitionList += [Transition(origState=states['S'][i],destState=states['E'][i],equation=lambdaName[i]+ '*' +states['S'][i] ,transitionType=TransitionType.T)]
        transitionList += [Transition(origState=states['E'][i],destState=states['I'][i],equation=' epsilon * ' +states['E'][i] ,transitionType=TransitionType.T)]
        transitionList += [Transition(origState=states['I'][i],destState=states['R'][i],equation=' gamma * ' +states['I'][i] ,transitionType=TransitionType.T)]
        for v in states:
            bdList += [Transition(origState=states[v][i], equation='d * '+states[v][i], transitionType=TransitionType.D)]
        bdList += [Transition(origState=states['S'][i], equation='d * '+N[i], transitionType=TransitionType.B)]
            
    ode = OperateOdeModel(stateList,
                          paramList,
                          derivedParamList=derivedParamList,
                          transitionList=transitionList,
                          birthDeathList=bdList)
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode


def Influenza_SLIARN(param=None):
    '''
    A simple influenza model from [1], page 323.

    .. math::
        \\frac{dS}{dt} &= -S \\beta (I + \\delta A) \\\\
        \\frac{dL}{dt} &= S \\beta (I + \\delta A) - \\kappa L \\\\
        \\frac{dI}{dt} &= p \\kappa L - \\alpha I \\\\
        \\frac{dA}{dt} &= (1-p) \\kappa L - \\eta A \\\\
        \\frac{dR}{dt} &= f \\alpha I + \\eta A \\\\ 
        \\frac{dN}{dt} &= -(1-f) \\alpha I
        
    References
    ----------
    .. [1] Mathematical Epidemiology, Lecture Notes in Mathematics,
           Brauer Fred, Springer 2008
    '''
    
    stateList = ['S', 'L','I','A','R','N']
    paramList = ['beta','p','kappa','alpha','f','delta','epsilon']
    odeList = [
               Transition(origState='S', equation='- beta * S * ( I + delta * A)',
               transitionType=TransitionType.ODE),
               Transition(origState='L', equation='beta * S * (I + delta * A) - kappa * L',
               transitionType=TransitionType.ODE),
               Transition(origState='I', equation='p * kappa * L - alpha * I',
               transitionType=TransitionType.ODE),
               Transition(origState='A', equation='(1-p) * kappa * L - epsilon * A',
               transitionType=TransitionType.ODE),
               Transition(origState='R', equation='f * alpha * I + epsilon * A',
               transitionType=TransitionType.ODE),
               Transition(origState='N', equation='-(1-f) * alpha * I',
               transitionType=TransitionType.ODE)
               ]
    # initialize the model
    ode = OperateOdeModel(stateList,
                          paramList,
                          odeList=odeList)
    
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def Legrand_Ebola_SEIHFR(param=None):
    '''
    The Legrand Ebola model with 6 compartments that includes the
    H = hospitalization and F = funeral state. Note that because this
    is an non-autonomous system, there are in fact a total of 7 states
    after conversion.  The set of equations that describes the model are

    .. math::
        \\frac{dS}{dt} &= -(\\beta_{I}SI + \\beta_{H}SH + \\beta_{F}SF) \\\\
        \\frac{dE}{dt} &= (\\beta_{I}SI + \\beta_{H}SH + \\beta_{F}SF) - \\alpha E \\\\
        \\frac{dI}{dt} &= \\alpha E - (\\gamma_{H} \\theta_{1} + \\gamma_{I}(1-\\theta_{1})(1-\\delta_{1}) + \gamma_{D}(1-\\theta_{1})\\delta_{1})I \\\\
        \\frac{dH}{dt} &= \\gamma_{H}\\theta_{1}I - (\\gamma_{DH}\\delta_{2} + \\gamma_{IH}(1-\\delta_{2}))H \\\\
        \\frac{dF}{dt} &= \\gamma_{D}(1-\\theta_{1})\\delta_{1}I + \\gamma_{DH}\\delta_{2}H - \\gamma_{F}F \\\\
        \\frac{dR}{dt} &= \\gamma_{I}(1-\\theta_{1})(1-\\delta_{1})I + \\gamma_{IH}(1-\\delta_{2})H + \\gamma_{F}F.

    References
    ----------
    .. [1] Understanding the dynamics of Ebola epidemics,
           Legrand J. et al. Epidemiology and Infection,
           Volume 135, Issue 4, pg 610-621, 2007

    Examples
    --------
    >>> x0 = [1.0, 3.0/200000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> t = numpy.linspace(0, 25, 100)
    >>> ode = common_models.Legrand_Ebola_SEIHFR([('beta_I',0.588),('beta_H',0.794),('beta_F',7.653),('omega_I',10.0/7.0),('omegaD',9.6/7.0),('omega_H',5.0/7.0),('omega_F',2.0/7.0),('alphaInv',7.0/7.0),('delta',0.81),('theta',0.80),('kappa',300.0),('interventionTime',7.0)]).setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''

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
    derivedParamList = [
        ('gamma_I', '1/omega_I'),
        ('gamma_D', '1/omega_D'),
        ('gamma_H', '1/omega_H'),
        ('gamma_F', '1/omega_F'),
        ('alpha', '1/alphaInv'),
        ('gamma_IH', '1/((1/gamma_I) - (1/gamma_H))'),
        ('gamma_DH', '1/((1/gamma_D) - (1/gamma_H))'),
        ('delta_1', 'delta * gamma_I / (delta * gamma_I + (1 - delta) * gamma_D)'),
        ('delta_2', 'delta * gamma_IH / (delta * gamma_IH + (1 - delta) * gamma_DH)'),
        ('theta_A', 'theta * (gamma_I * (1 - delta_1) + gamma_D * delta_1)'),
        ('theta_1', 'theta_A/ (theta_A +  (1 - theta) * gamma_H)'),
        ('beta_H_Time', 'beta_H * (1 - (1/ (1+exp(-kappa*(tau-interventionTime)))))'),
        ('beta_F_Time', 'beta_F * (1 - (1/ (1+exp(-kappa*(tau-interventionTime)))))')
        ]

    # alternatively, we can do it on the operate ode model
    ode = OperateOdeModel(state, params)
    # add the derived parameter
    ode.setDerivedParamList(derivedParamList)

    # define the set of transitions
    # name of origin state
    # name of target state
    # equation
    # type of equation, which is a transition between two state in this case

    transitionList = [
        Transition(origState='S', destState='E',
                   equation='(beta_I * S * I + beta_H_Time * S * H + beta_F_Time * S * F)',
                   transitionType=TransitionType.T),
        Transition(origState='E', destState='I',
                   equation='alpha * E',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='H',
                   equation='gamma_H * theta_1 * I',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='F',
                   equation='gamma_D * (1 - theta_1) * delta_1 * I',
                   transitionType=TransitionType.T),
        Transition(origState='I', destState='R',
                   equation='gamma_I * (1 - theta_1) * (1 - delta_1) * I',
                   transitionType=TransitionType.T),
        Transition(origState='H', destState='F',
                   equation='gamma_DH * delta_2 * H',
                   transitionType=TransitionType.T),
        Transition(origState='H', destState='R',
                   equation='gamma_IH * (1 - delta_2) * H',
                   transitionType=TransitionType.T),
        Transition(origState='F', destState='R',
                   equation='gamma_F * F',
                   transitionType=TransitionType.T)
        ]
    #print transitionList
    bdList = [Transition(origState='tau', equation='1',
                         transitionType=TransitionType.B)]

    # see how we can insert the transitions later, after initializing the ode object
    # this is not the preferred choice though
    ode.setTransitionList(transitionList)
    ode.setBirthDeathList(bdList)
    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def Lotka_Volterra(param=None):
    '''
    Standard Lotka-Volterra model with two states and four parameters

    .. math::
        \\frac{dx}{dt} &= \\alpha x - cxy \\\\
        \\frac{dy}{dt} &= -\\delta y + \\gamma xy

    References
    ----------
    .. [1] Analytical Note on Certain Rhythmic Relations in Organic Systems,
           Lotka Alfred J., Proceedings of the National Academy of Sciences of the
           United States of America, Volume 7, Issue 7, pg. 410-415, 1920.

    Examples
    --------

    >>> params = {'alpha':1,'delta':3,'c':2,'gamma':6}
    >>> ode = common_models.Lotka_Volterra(params).setInitialValue([2.0,6.0],0)
    >>> t = numpy.linspace(0.1,100,10000)
    >>> ode.integrate(t)
    >>> ode.plot()

    '''

    # our two state and four parameters
    # no idea why they are not in capital
    stateList = ['x', 'y']
    # while these 4 are
    paramList = ['alpha', 'delta', 'c', 'gamma']
    # then define the set of ode
    odeList = [
        Transition(origState='x', equation='alpha * x - c * x * y',
                   transitionType=TransitionType.ODE),
        Transition(origState='y', equation='-delta * y + gamma * x * y',
                   transitionType=TransitionType.ODE)
        ]

    ode = OperateOdeModel(stateList,
                          paramList,
                          odeList=odeList)
    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def Lotka_Volterra_4State(param=None):
    '''
    The four state Lotka-Volterra model. A common interpretation is that
    a = Grass, x = rabbits, y = foxes and b is the death of foxes.

    .. math::
        \\frac{da}{dt} &= k_{0} a x \\\\
        \\frac{dx}{dt} &= k_{0} a x - k_{1} x y \\\\
        \\frac{dy}{dt} &= k_{1} x y - k_{2} y \\\\
        \\frac{db}{dt} &= k_{2} y

    References
    ----------
    .. [1] Analytical Note on Certain Rhythmic Relations in Organic Systems,
           Lotka Alfred J., Proceedings of the National Academy of Sciences of the
           United States of America, Volume 7, Issue 7, pg. 410-415, 1920.

    Examples
    --------

    >>> x0 = [150.0, 10.0, 10.0, 0.0]
    >>> t = numpy.linspace(0,15,100)
    >>> params = [0.01,0.1,1.0]
    >>> ode = common_models.Lotka_Volterra_4State(params).setInitialValue(x0,t[0])
    >>> ode.integrate(t[1::])
    >>> ode.plot()

    '''

    # four states
    stateList = ['a', 'x', 'y', 'b']
    # three parameters
    paramList = ['k0', 'k1', 'k2']

    # then define the set of ode
    transitionList = [
        Transition(origState='a', destState='x',
                   equation='k0 * a * x',
                   transitionType=TransitionType.T),
        Transition(origState='x', destState='y',
                   equation='k1 * x * y',
                   transitionType=TransitionType.T),
        Transition(origState='y', destState='b',
                   equation='k2 * y',
                   transitionType=TransitionType.T)
        ]

    ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)

    # set return, depending on whether we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def FitzHugh(param=None):
    '''
    The standard FitzHugh model without external input

    .. math::
        \\frac{dV}{dt} &=  c ( V - \\frac{V^{3}}{3} + R) \\\\
        \\frac{dR}{dt} &= -\\frac{1}{c}(V - a + bR).

    References
    ----------
    .. [1] Impulses and Physiological States in Theoretical Models of Nerve Membrane,
           Biophysical Journal, FitzHugh Richard, Volume 1, Issue 6, pg. 445-466, 1961.

    Examples
    --------

    >>> ode = common_models.FitzHugh({'a':0.2,'b':0.2,'c':3.0})
    >>> t = numpy.linspace(0,20,101)
    >>> x0 = [1.0,-1.0]
    >>> ode.setInitialValue(x0,t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()

    '''

    # the two states
    stateList = ['V', 'R']
    # and the three parameters
    paramList = ['a', 'b', 'c']

    # the set of ode
    odeList = [
        Transition(origState='V', equation='c * (V - (V * V * V)/3 + R)',
                   transitionType=TransitionType.ODE),
        Transition(origState='R', equation='-( (V - a + b * R)/c )',
                   transitionType=TransitionType.ODE)
        ]
    # setup our ode
    ode = OperateOdeModel(stateList,
                          paramList,
                          derivedParamList=None,
                          transitionList=None,
                          birthDeathList=None,
                          odeList=odeList)
    # set return, depending on whehter we have input the parameters
    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def Lorenz(param=None):
    '''
    Lorenz attractor define by three parameters, :math:`\\beta,\\sigma,\\rho`

    .. math::
        \\frac{dx}{dt} &= \\sigma (y-x) \\\\
        \\frac{dy}{dt} &= x (\\rho - z) - y \\\\
        \\frac{dz}{dt} &= xy - \\beta z

    References
    ----------
    .. [1] Deterministic Nonperiodic Flow, Lorenz, Edward N.,
           Journal of the Atmospheric Sciences,
           Volume 20, Issus 2, pgs 130-141, 1963

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> t = numpy.linspace(0,20,101)
    >>> params = {'beta':8.0/3.0,'sigma':10.0,'rho':28.0}
    >>> ode = common_models.Lorenze(params).setInitialValue([1.,1.,1.],t[0])
    >>> solution = ode.integrate(t[1::])
    >>> plt.plot(solution[:,0],solution[:,2])
    >>> plt.show()

    '''

    stateList = ['x', 'y', 'z']
    paramList = ['beta', 'sigma', 'rho']
    odeList = [
        Transition(origState='x', equation='sigma * (y - x)',
                   transitionType=TransitionType.ODE),
        Transition(origState='y', equation='x * (rho - z) - y',
                   transitionType=TransitionType.ODE),
        Transition(origState='z', equation='x * y - beta * z',
                   transitionType=TransitionType.ODE)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList, paramList, odeList=odeList)

    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def vanDelPol(param=None):
    '''
    The van der Pol equation, a second order ode

    .. math::
        y^{\prime\prime} - \mu (1-y^{2}) y^{\prime} + y = 0

    where :math:`\mu > 0`.  This can be converted to a first order ode by equating :math:`x = y^{\prime}`

    .. math::
        x^{\prime} - \mu (1 - y^{2}) x + y = 0

    which result in a coupled ode

    .. math::
        x^{\\prime} &= \\mu (1 - y^{2}) x - y \\\\
        y^{\\prime} &= x

    and this can be solved via standard method

    References
    ----------
    .. [1] On Relaxed Oscillations, van der Pol, Balthasar,
           The London, Edinburgh, and Dublin Philosophical Magazine 
           and Journal of Science,
           Volume 2, Issue 11, pg.  978-992, 1926

    Examples
    --------

    >>> from odeModel import common_models
    >>> import numpy
    >>> t = numpy.linspace(0,20,1000)
    >>> ode = common_models.vanDelPol({'mu':1.0}).setInitialValue([2.0,0.0],t[0])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot()
    '''

    stateList = ['y', 'x']
    paramList = ['mu']
    odeList = [
        Transition(origState='y', equation='x',
                   transitionType=TransitionType.ODE),
        Transition(origState='x', equation='mu * (1-y*y) * x -  y',
                   transitionType=TransitionType.ODE)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList, paramList, odeList=odeList)

    if param is None:
        return ode
    else:
        ode.setParameters(param)
        return ode

def Robertson(param=None):
    '''
    The so called Robertson problem, which is a standard example used to test stiff integrator.

    .. math::
        \\frac{dy_{1}}{dt} &= -0.04 y_{1} + 1 \cdot 10^{4} y_{2} y_{3} \\\\
        \\frac{dy_{2}}{dt} &= 0.04 y_{1} - 1 \cdot 10^{4} y_{2} y_{3} - 3 \cdot 10^{7} y_{2}^{2}\\\\
        \\frac{dy_{3}}{dt} &= 3 \cdot 10^{7} y_{2}^{2}

    References
    ----------
    .. [1] The solution of a set of reaction rate equations, Robertson, H.H.,
           pg. 178-182, Academic Press, 1966

    Examples
    --------

    >>> from odeModel import common_models
    >>> import numpy
    >>> t = numpy.append(0,4*numpy.logspace(-6,6,1000))
    >>> ode = common_models.Robertson().setInitialValue([1.0,0.0,0.0],t[0]).integrate(t[1::])
    >>> solution = ode.integrate(t[1::])
    >>> ode.plot() # note that this is not being plotted in the log scale

    '''
    # note how we have short handed the definition
    stateList = ['y1:4']
    # note that we do not have any parameters, or rather,
    # we have hard coded in the parameters
    paramList = []
    transitionList = [
        Transition(origState='y1', destState='y2',
                   equation='0.04*y1',
                   transitionType=TransitionType.T),
        Transition(origState='y2', destState='y1',
                   equation='1e4 * y2 * y3',
                   transitionType=TransitionType.T),
        Transition(origState='y2', destState='y3',
                   equation='3e7 * y2 * y2',
                   transitionType=TransitionType.T)
        ]
    # initialize the model
    ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)

    if param is None:
        return ode
    else:
        raise Warning("Input parameters not used")
        return ode

