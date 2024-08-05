import numpy as np
cimport numpy as np

cimport scipy.special.cython_special as csc
from libc.math cimport floor
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate the div 0 error checking
def _cy_test_tau_leap_safety(np.ndarray[np.float64_t] x,
                             np.ndarray[np.int64_t, ndim=2] reactant_mat,
                             np.ndarray[np.float64_t] rates,
                             double tau_scale,
                             double epsilon):
    """
    Additional safety test on :math:`\\tau`-leap, decrease the step size if
    the original is not small enough.  Decrease a couple of times and then
    bail out because we don't want to spend too long decreasing the
    step size until we find a suitable one.
    """
    #view on arrays
    cdef double[:] rates_view = rates
    cdef np.int64_t n_rates = rates.shape[0]
    cdef double[:] x_view = x
    cdef np.int64_t[:, :] reactant_mat_view = reactant_mat
    cdef np.int64_t n_reactants = reactant_mat.shape[0]

    cdef double mu, max_cdf, new_cdf
    cdef double total_rate = np.sum(rates)
    safe = False  # when True, indicates that tau_scale is sufficiently small
    cdef int count = 0  # number of attempts to find acceptable tau_scale
    while safe is False:
        # print(count)
        # print(tau_scale)
        cdf_val = 1.0
        for i in range(n_rates):  # loop over transitions, i
            for j in range(n_reactants):  # loop over states, j
                if reactant_mat_view[j, i] == 1:  # is state j involved in transition i?
                    mu = rates_view[i] * tau_scale  # expected number of events for transition i
                    new_cdf = csc.pdtr(floor(x_view[j]), mu)  # prob transitions of order state pop
                    if new_cdf < cdf_val:
                        cdf_val = new_cdf
            # if new_cdf < cdf_val:  # think this extra condition is redundant
            #     cdf_val = new_cdf
            #cdf_val[i * reactant_mat.shape[0] : (i * reactant_mat.shape[0]) + len(rates)] = _ppois(xi, mu=tau_scale*r)

        # the expected probability that our jump will exceed the value
        max_cdf = 1.0 - cdf_val
        # print(max_cdf)
        # cannot allow it to exceed out epsilon
        if max_cdf > epsilon:
            tau_scale /= (max_cdf / epsilon)
        else:
            safe = True

        if count > 256:
            print("count error")
            return False
        
        # if tau_scale*total_rate <= 1.0:  # leave out Gillespie regime catch for now
        #     print("scale error")
        #     return False
        count += 1

    return tau_scale, True
