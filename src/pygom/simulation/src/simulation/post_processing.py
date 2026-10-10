"""
Post processing of stochastic outputs.

The time points required by the user, t_eval, might not match up to the output times.
For TimeSeriesResult outputs (from tau leap) we use linear interpolation. For EventSeriesResult
outputs (from exact simulation) we extract the system snapshot.

NOTE: This may create non-integer values for the TimeSeriesResult, in later versions
we should ensure that the algorithm hits t_eval values during solving phase.
"""

import numpy as np

def extract_state_at_target_times(t, y, target_time):
    """
    Given state values in event based (e.g. exact) system, extract the exact
    states at each target time.
    
    Parameters
    ----------
    t : numpy.ndarray
        Event times (plus initial time, which does not involve an event)
    y : numpy.ndarray
        State values which exist between consecutive timepoints
    target_time : numpy.ndarray
        Target timesteps

    Returns
    -------
    numpy.ndarray
    """
    # Find insertion positions
    idx = np.searchsorted(t, target_time, side="right") - 1

    # Clamp to range [0, len(t)-1]
    idx = np.clip(idx, 0, len(t)-1)

    return y[idx]

def interpolate_state_at_times(t, y, target_time):
    """
    Given state values in time based (e.g. tau leap) system, interpolate the
    states at each target time.
    
    Parameters
    ----------
    t : numpy.ndarray
        Timepoints
    y : numpy.ndarray
        State values at each timepoint
    target_time : numpy.ndarray
        Target timesteps

    Returns
    -------
    numpy.ndarray
    """
    n_state = y.shape[1]
    y_interp = np.zeros((len(target_time), n_state))  # empty matrix to receive scaled data = (new timepoints x vars)
    # linearly interpolate to new timepoints
    for i in range(n_state):
        y_interp[:, i]=np.interp(target_time, t, y[:,i])

    return y_interp

def bin_events(event_ids, event_times, target_times, n_events):
    """
    Given event ids and times, evaluate how many of each occur between the target times
    
    Parameters
    ----------
    event_ids : numpy.ndarray
        index of event which occured
    event_times : numpy.ndarray
        Times at which events occured
    target_time : numpy.ndarray
        Target timesteps
    n_events: int
        Number of possible events

    Returns
    -------
    numpy.ndarray
    """

    n_bins = len(target_times) - 1

    # which time bin each event belongs to
    time_bins = np.searchsorted(target_times, event_times, side="right") - 1

    # keep only events inside the intervals
    mask = (time_bins >= 0) & (time_bins < n_bins)

    # accumulate counts
    counts = np.zeros((n_bins, n_events), dtype=int)
    np.add.at(counts, (time_bins[mask], event_ids[mask]), 1)

    return counts

def change_bins(event_counts, old_breaks, new_breaks):
    """
    Extrapolate the number of events occuring within the old_breaks to a new set new_breaks.
    Warning: This can create non-integer values.
    
    Parameters
    ----------
    event_counts : numpy.ndarray
        Matrix of number of times each event occurs per timestep
    old_breaks : numpy.ndarray
        Timepoints of event_counts
    new_breaks : numpy.ndarray
        Target timesteps

    Returns
    -------
    numpy.ndarray
    """

    n_state = event_counts.shape[1]
    events_per_bin = np.zeros((len(new_breaks)-1, n_state))  # empty matrix to receive scaled data = (new timepoints x vars)

    for i in range(n_state):
        tmin_old = old_breaks[:-1]
        tmax_old = old_breaks[1:]

        tmin_new = new_breaks[:-1]
        tmax_new = new_breaks[1:]

        dt = tmax_old-tmin_old

        # density = np.divide(event_counts[:, i], dt, where=dt>0)

        density = np.divide(
            event_counts[:, i],
            dt,
            out=np.zeros_like(dt, dtype=float),
            where=dt > 0,
        )


        # Broadcast to compute overlaps
        overlap_left = np.maximum(tmin_old[:, None], tmin_new[None, :])
        overlap_right = np.minimum(tmax_old[:, None], tmax_new[None, :])
        overlap = np.maximum(0, overlap_right-overlap_left)

        events_per_bin[:, i] = (density[:, None] * overlap).sum(axis=0)

    return events_per_bin
