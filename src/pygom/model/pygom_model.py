from .model_spec import ModelSpec

from .ode_utils.compiler import CompileCode

from .ode_utils.maths_methods_store import MathsMethodsRegistry

import logging
import warnings


from ._model_errors import InputError
from ..simulation.src.simulation.stochastic.solve_stochastic import solve_stochastic
from ..simulation.src.simulation.deterministic.solve_deterministic import solve_deterministic
from .ode_variable import CallableParameter

import numpy as np
from numbers import Number



class DependencyManager:
    """
    Individual stores take care of their own caches for the most part,
    but sometimes dependency across different stores can lead to
    cache invalidation.

    Currently this deals with the simple case: adding new events invalidates
    the maths methods store. For now, the dependency manager is overly
    complicated, but ready if required for more complex model inter-
    dependency in the future.

    TODO: is it this simple? adding states and params can also change the
          math obects, simply because an n_param x n_state matrix will need
          more rows and columns, even if the extra params do nothing
    """
    def __init__(self):

        self._dependency_map = {
            "events": ["maths_methods"]
        }

        self._dirty = {
            "maths_methods": True
        }

    def mark_dirty(self, dependency):
        self._dirty[dependency] = True

    def mark_clean(self, dependency):
        self._dirty[dependency] = False

    def is_dirty(self, component):
        return self._dirty[component]

    def notify_change(self, change_location):
        """
        Notify all relevant dependencies that a change has just occurred.
        """
        if change_location not in self._dependency_map.keys():
            return ValueError(f"Feature, '{change_location}', is not recognised as a model component")

        dependency_locations = self._dependency_map[change_location]

        for dependency_location in dependency_locations:
            self.mark_dirty(dependency_location)

class Model():

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            event=None,
            backend='lambda'
        ):

        ## Dependency manager ##
        self._DependencyManager = DependencyManager()

        ## Foundational model specifications ##

        # (states, params, events)
        self._ModelSpec = ModelSpec(
            state=state,
            param=param,
            derived_param=derived_param,
            event=event,
            dependency_manager=self._DependencyManager
        )

        # Compiler
        self._SC = CompileCode(backend=backend)

        ## Maths methods registry ##
        self._MethodRegister = MathsMethodsRegistry(
            model_spec=self._ModelSpec,
            compiler=self._SC,
            dependency_manager=self._DependencyManager
        )

        for method in self._MethodRegister._maths_methods:
            name = method.method_name
            setattr(self, name, self._MethodRegister[name])



    ## Simulation interface ##

    def solve_deterministic(
            self,
            t,
            method="LSODA",
            dense_output=False,
            events=None,
            vectorized=False,
            args=None,
            perf=False,
            iteration=1,
            seed=None,
            parallel=False,
            **options):
    
        if (not self._ModelSpec.has_stochastic_parameters) and (iteration>1):
            warnings.warn("System only has deterministic parameters, but multiple"
                          "iterations specified. Perhaps you wish to set iteration=1 instead?")

        # Normalise time input
        t0 = self.initial_time
        if isinstance(t, Number):
            t_span = (t0, t)
            t_eval = None
        elif isinstance(t, np.ndarray):
            t_span = (t0, t.max())
            t_eval = t
        else:
            raise InputError("Unknown dtype for time. Expected Number or numpy.array")
        

        def _run_single_iteration(iter_ss):
            # TODO: fix stohastic params
            # params_ss = iter_ss.spawn(1)[0]
            # stochastic_params = self._ModelSpec.stochastic_parameters
            # param_seqs = params_ss.spawn(len(stochastic_params))

            # for seq, param in zip(param_seqs, stochastic_params.values()):
            #     if isinstance(param.value, CallableParameter):
            #         param.value.rng = np.random.default_rng(seq)

            # self._ModelSpec.new_parameter_realisation()

            jacobian_methods = ['LSODA', 'Radau', 'BDF']

            if method in jacobian_methods:
                return solve_deterministic(
                        ode_eqns=self._MethodRegister.ode.T,
                        n_state=self._ModelSpec.num_state,
                        event_rates=self._MethodRegister.event_rate_vector.T,
                        n_event=self._ModelSpec.num_event,
                        t_span=t_span,
                        y0=self.initial_states,
                        jac_ode=self._MethodRegister.ode_jacobian_states.T,
                        jac_events=self._MethodRegister.event_rates_jacobian_states.T,
                        method=method,
                        t_eval=t_eval,
                        dense_output=dense_output,
                        events=events,
                        vectorized=vectorized,
                        args=args,
                        perf=perf,
                        **options)
            else:
                return solve_deterministic(
                        ode_eqns=self._MethodRegister.ode.T,
                        n_state=self._ModelSpec.num_state,
                        event_rates=self._MethodRegister.event_rate_vector.T,
                        n_event=self._ModelSpec.num_event,
                        t_span=t_span,
                        y0=self.initial_states,
                        method=method,
                        t_eval=t_eval,
                        dense_output=dense_output,
                        events=events,
                        vectorized=vectorized,
                        args=args,
                        perf=perf,
                        **options)
        
        master_ss = np.random.SeedSequence(seed)
        iteration_seqs = master_ss.spawn(iteration)

        if not parallel:
            logging.debug("Performing serial simulation")

            out = []
            for i in range(iteration):
                result = _run_single_iteration(iteration_seqs[i])
                out.append(result)

            return out

    def solve_stochastic(
            self,
            t,
            method,
            seed=None,
            proceed_if_rates_zero=False,
            perf=False,
            iteration=1,
            parallel=False,
            client=None,
            parallel_backend="dask",
            **options):
        
        y_min = np.zeros(self.num_state, dtype=int)
        y_max = np.full(self.num_state, np.inf)

        if method == "cao2006":
            options["transition_mean_func"] = self.transition_mean.T
            options["transition_var_func"] = self.transition_variance.T

        # Normalise time input
        t0 = self.initial_time
        if isinstance(t, Number):
            t_span = (t0, t)
            t_eval = None
        elif isinstance(t, np.ndarray):
            t_span = (t0, t.max())
            t_eval = t
        else:
            raise InputError("Unknown dtype for time. Expected Number or numpy.array")
        
        # Evaluate stoichiometry_matrix at initial conditions and assume constant
        stoichiometry_matrix = self.state_change_matrix(self.initial_state, self.initial_time)

        def _run_single_iteration(iter_ss):
            solver_ss, params_ss = iter_ss.spawn(2)
            solver_rng = np.random.default_rng(solver_ss)
            stochastic_params = self._parameter_store.stochastic_parameters
            param_seqs = params_ss.spawn(len(stochastic_params))

            for seq, param in zip(param_seqs, stochastic_params.values()):
                if isinstance(param.value, CallableParameter):
                    param.value.rng = np.random.default_rng(seq)

            self._parameter_store.new_realisation()

            return solve_stochastic(
                    event_rates=self.event_rate_vector.T,
                    stoichiometry_matrix=stoichiometry_matrix,
                    t_span=t_span,
                    y0=self.initial_state,
                    y_min=y_min,
                    y_max=y_max,
                    method=method,
                    t_eval=t_eval,
                    proceed_if_rates_zero=proceed_if_rates_zero,
                    perf=perf,
                    rng=solver_rng,
                    **options)
        
        master_ss = np.random.SeedSequence(seed)
        iteration_seqs = master_ss.spawn(iteration)

        if not parallel:
            logging.debug("Performing serial simulation")

            out = []
            for i in range(iteration):
                # model = self.copy()
                result = _run_single_iteration(iteration_seqs[i])
                out.append(result)

            return out

        # ----------------------------------------------------
        # Parallel execution
        # ----------------------------------------------------
        
        if parallel_backend == "dask":
            logging.debug(f"Using {parallel_backend} for parallel simulation")

            if client is None:
                from dask.distributed import Client
                client = Client()
            
            futures = [
                client.submit(_run_single_iteration, iteration_seqs[i], self.copy())
                for i in range(iteration)
            ]
            return client.gather(futures)

        raise ValueError(f"Unknown backend: {parallel_backend}")


    ##.....Loss functions interface.....##

    ##.....ABC interface.....##



    ## User interface ##

    @property
    def parameters(self):
        """
        Get a dict of {str_id: numeric value}

        NOTE: can be anything we want from model_spec (e.g. list)
        """
        return self._ModelSpec.parameter_id_value_dict

    @parameters.setter
    def parameters(self, parameters):
        self._ModelSpec.set_parameter_values(parameters)


    @property
    def initial_states(self):
        """
        Get a dict of {str_id: numeric value}
        """
        return self._ModelSpec.initial_state_list

    @initial_states.setter
    def initial_states(self, ics):
        self._ModelSpec.set_initial_states(ics)

    @property
    def initial_time(self):
        """
        Initial time value
        """
        return self._ModelSpec.initial_time

    @initial_time.setter
    def initial_time(self, t0):
        self._ModelSpec.set_initial_time(t0)

    # TODO:
    # 1) Transfer remaining mathmethods from DeterminsiticODE
    # 2) random number seeds