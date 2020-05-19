import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY
cimport cython

from pygom.utilR.distn import test_seed

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate the div 0 error checking
def _cy_checkJump(x, new_x, t, jump_time):

    # Static typing of the input arguments leads to slowdown
    # I think because the input objects are not touched directly by the function
    # (Only memory view is touched in practice, until return statement)
    # Function takes 14% runtime of python original

    cdef double[:] my_view = new_x
    cdef int i
	
    N_x = my_view.shape[0]
    
    failed_jump = False
    
    for i in range(N_x):
        if my_view[i] < 0:
            failed_jump = True

    if failed_jump:
        # print "Illegal jump, x: %s, new x: %s" % (x, new_x)
        return x, t, False
    else:
        t += jump_time
        return new_x, t, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate the div 0 error checking
def _cy_newJumpTimes(np.ndarray[np.float64_t] rates, seed=None):
    """
    Generate the new jump times assuming that the rates follow an exponential
    distribution
    """
    cdef int  Nx
    cdef double[:] my_view = rates
    Nx = my_view.shape[0]
    
    #tau = np.empty([Nx])
    cdef np.ndarray[np.float64_t] my_view_2 = np.empty([Nx], dtype=np.float64)
    #cdef double[:] my_view_2 = tau

    cdef int i
    for i in range(0, Nx):
        if my_view[i] > 0:
            if seed is None:
                my_view_2[i] = np.random.exponential(scale=1.0/my_view[i], size=1)[0]
            else:
                my_view_2[i] = test_seed(seed).exponential(scale=1.0/my_view[i], size=1)[0]
        else:
            my_view_2[i] = INFINITY
    
    return my_view_2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate the div 0 error checking
def _cy_updateStateWithJump(x, transition_index, state_change_mat, n=1.0):
    """
    Updates the states given a jump.  Makes use the state change
    matrix, and updates according to the number of times this
    transition has happened
    """
    return x + state_change_mat[:, transition_index]*n

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate the div 0 error checking
def _cy_firstReaction(np.ndarray[np.float64_t] x_in, 
                      double t, 
                      np.ndarray[np.int64_t, ndim=2] state_change_mat,
                      np.ndarray[np.float64_t] rates,
                      seed):
     
    # This cython function is not the complete _firstReaction
    # It has different input argumets to the python-level function

    cdef double[:] x = x_in

    # find our jump times
    # inline _cy_newJumpTimes
    cdef double[:] my_view = rates
    
    #cdef int  Nx
    #Nx = my_view.shape[0]
    Nx = rates.shape[0]

    #jump_times = np.empty([Nx])
    cdef np.ndarray[np.float64_t] my_view_2 = np.empty([Nx], dtype=np.float64)

    cdef int i
    for i in range(0, Nx):
        if my_view[i] > 0:
            if seed is None:
                my_view_2[i] = np.random.exponential(scale=1.0/my_view[i], size=1)[0]
            else:
                my_view_2[i] = test_seed(seed).exponential(scale=1.0/my_view[i], size=1)[0]
        else:
            my_view_2[i] = INFINITY


    all_inf = True    
    for i in range(0, Nx):
       if my_view_2[i] != INFINITY:
           all_inf = False

    if all_inf:
        return x_in, t, False

    # first jump

    cdef np.float64_t smallest = INFINITY
    cdef long min_index

    for i in range(0, Nx):
        if my_view_2[i] < smallest:
            smallest = my_view_2[i]
            min_index = i
	   
    x_shape = x_in.shape[0]
    new_x = np.empty([x_shape])

    cdef double[:] new_x_view = new_x
    
    # Inline _cy_updateStateWithJump

    cdef np.int64_t[:,:] my_view_4 = state_change_mat
    for i in range(0, x_shape):
        new_x_view[i] = x[i] + my_view_4[i, min_index]

    # Inline _cy_checkJump
  
    Nx = new_x_view.shape[0]
     
    failed_jump = False
    
    for i in range(Nx):
        if new_x_view[i] < 0:
            failed_jump = True

    if failed_jump:
        return_x, return_t, return_sucess = x_in, t, False
    else:
        #t += jump_times[min_index]
        t += my_view_2[min_index]
        return_x, return_t, return_sucess =  new_x, t, True
   
    return return_x, return_t, return_sucess 
