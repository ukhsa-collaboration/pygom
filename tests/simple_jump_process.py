
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

from pygom import Transition, TransitionType, SimulateOde
import numpy as np

# Setup logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.DEBUG)


# construct model
states = ['S', 'I', 'R']
params = ['beta', 'gamma', 'N']
transitions = [Transition(origin='S', destination='I', equation='beta*S*I/N',
                          transition_type=TransitionType.T),
               Transition(origin='I', destination='R', equation='gamma*I',
                          transition_type=TransitionType.T)]

# initial conditions
N = 7781984.0
in_inf = round(0.0000001*N)
init_state = [N - in_inf, in_inf, 0.0]
#
# # time
max_t = 9 # 50
t = np.linspace (0 , max_t , 101)
#
# # deterministic parameter values
param_evals = [('beta', 3.6), ('gamma', 0.2), ('N', N)]

# construct model
model_j = SimulateOde(states, params, transition=transitions)
model_j.parameters = param_evals
model_j.initial_values = (init_state, t[0])


# run 10 simulations
start = time.time()
simX, simT = model_j.simulate_jump(t[1::], iteration=10, full_output=True)
end = time.time()

logging.info('Simulation took {} seconds'.format(end - start))
