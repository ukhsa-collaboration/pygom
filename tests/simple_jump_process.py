
# coding: utf-8

# # Stochastic simulation
#
# Examples taken from https://arxiv.org/pdf/1803.06934.pdf (see page 11 for stochastic simulations).
#
# Examples are performed on an SIR model.
#
# $\frac{dS}{dt} = -\beta S I $
#
# $\frac{dI}{dt} = \beta S I - \gamma I$
#
# $\frac{dR}{dt} = \gamma I$

# In[1]:

import logging
import time

import pygom
import pkg_resources
print('PyGOM version %s' %pkg_resources.get_distribution('pygom').version)

from pygom import Transition, TransitionType, Event, SimulateOde
import numpy as np

# Setup logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.DEBUG)


# construct model
state_list = [('S', (0,None)), ('I', (0, None)), ('R', (0, None))]
param_list = ['beta', 'gamma', 'N']

trans_inf=Transition(origin='S', destination='I',transition_type=TransitionType.T)
event_inf=Event(rate='beta*S*I/N', transition_list=[trans_inf])

trans_rec=Transition(origin='I', destination='R',transition_type=TransitionType.T)
event_rec=Event(rate='gamma*I', transition_list=[trans_rec])

model = SimulateOde(state=state_list,
                    param=param_list,
                    event=[event_inf, event_rec])

# initial conditions
N = 7781984.0                               # not sure why this number was chosen initially...
i0 = 10                                     # try to avoid stochastic extinction
init_state = [N - i0, i0, 0.0]

# time
max_t = 9 # 50
t = np.linspace (0 , max_t , 101)

# deterministic parameter values
param_evals = [('beta', 3.6), ('gamma', 0.2), ('N', N)]     # R0=18

# add params and ICs to model
model.parameters = param_evals
model.initial_values = (init_state, t[0])

# run 10 simulations
N_ITERATION=10
start = time.time()
solution, simJump, simT = model.solve_stochast(t, iteration=N_ITERATION, full_output=True)
end = time.time()

logging.info('Simulation took {} seconds'.format(end - start))
