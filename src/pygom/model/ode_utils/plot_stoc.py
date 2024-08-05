'''
Created on 14 Jan 2019

@author: thomas.finnie
'''
import numpy as np

import matplotlib.pyplot

def get_rows_cols(n):
    '''
    Calculates a pleasing layout for subplots
    '''
    rows_and_cols = {1: (1, 1),
                     2: (1, 2),
                     3: (1, 3),
                     4: (2, 2)
                    }
    return rows_and_cols.get(n, (int(np.ceil(n / 3)), 3))

def get_subplot_num(n, rows, cols):
    '''
    Given a particular number of columns work out where a plot will go
    '''
    if rows == 1:
        return (n)
    return (int(np.ceil(n / cols)) - 1, #get the row
            (((n - 1) % cols) + 1) - 1) #get the col

def plot_state(sim_x,
               sim_t,
               sub_plot,
               i,
               state_name,
               palette
               ):
    '''
    Produces an individual plot
    '''
    for num, realisation in enumerate(sim_x):
        sub_plot.plot(sim_t, realisation[:,i],
                               color=palette(num),
                               linewidth=1,
                               alpha=0.9)
        sub_plot.set_title(state_name)

def plot_stoc(solution, t, stochastic_model):
    '''
    Plot the results of a stocastic simulation

    Parameters
    ==========
    solution: :class: list
        results of the stochastic simulation
    t: array like
        the vector of time where the integration output correspond to
    stochastic_model: :class: `pygom.SimulateOde`
        the model from which this simulation was generated

    Notes
     -----
    If we have 5 states or more, it will always be arrange such
    that it has 3 columns.
    '''
    #TODO: lots of checks

    #some basic information
    num_states = solution[0][0].shape[0]

    # create a color palette
    palette = matplotlib.pyplot.get_cmap('Set1')

    #get the rows and cols
    rows, cols = get_rows_cols(num_states)
    #get the subplots
    f, axarr = matplotlib.pyplot.subplots(rows, cols)

    #loop over states and build the plots
    for i, state in enumerate(stochastic_model._stateList):
        plot_state(sim_x=solution,
                   sim_t=t,
                   sub_plot=axarr[get_subplot_num(i, rows=rows, cols=cols)],
                   i=i,
                   state_name=state,
                   palette=palette)
    #Tidy-up and plot out
    f.tight_layout()
    matplotlib.pyplot.show()
