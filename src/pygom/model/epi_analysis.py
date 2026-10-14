import sympy
from sympy import Matrix
from .maths.mathsmethod import NumericMethod
from .transition import TransitionType

class Equilibrium(NumericMethod):
    method_name = "equilibrium"
    depends_on = ["ode"]

    def _build_expression(self):

        # Subset to just the dynamic states. If we have a cumulative compartment
        # then it will appear that the system never has an equilibirum (S, I R may
        # find equilirbium, but D forever grows)
        
        dynamic_state_ids = self._model_spec.reject_states_by_tag("cumulative")

        dynamic_state_indices = [
            self._model_spec.get_state_index(state)
            for state in dynamic_state_ids
        ]

        dynamic_state_symbols = [
            self._model_spec.state_symbol_list[i] for i in dynamic_state_indices
        ]

        ode = self._method_register["ode"].get_equation()
        dynamic_ode = ode[dynamic_state_indices, :]

        # Include deirved_parameters as constraint?

        constraints = Matrix([
            symb - expr
            for symb, expr in self._model_spec.derived_parameter_symbol_expression_dict.items()
        ])

        eqns = Matrix.vstack(
            dynamic_ode,
            constraints
        )

        self._sympy_expression = sympy.solve(
            eqns,
            dynamic_state_symbols,
            dict=True
        )

class DiseaseFreeEquilibrium(NumericMethod):
    method_name = "disease_free_equilibrium"
    depends_on = ["equilibrium"]

    def _build_expression(self):

        infected_state_ids = self._model_spec.get_states_by_tag("infected")

        infected_state_symbols = [
            self._model_spec.state_dict[id] for id in infected_state_ids
        ]

        equilibria = self._method_register["equilibrium"].get_equation()

        self._sympy_expression = [
            eq
            for eq in equilibria
            if all(
                sympy.simplify(
                    eq.get(state, 0)
                ) == 0
                for state in infected_state_symbols
            )
        ]

class InfectionVector(NumericMethod):
    method_name = "infection_vector"
    depends_on = [
        "event_rate_vector",
        "state_change_matrix"
    ]

    """
    Rates at which new infections occur
    """

    def _build_expression(self):

        infected_state_ids = self._model_spec.get_states_by_tag("infected")

        event_rate_vector = self._method_register["event_rate_vector"].get_equation()

        # container for output:
        F = sympy.zeros(len(infected_state_ids), 1)

        for event_i, event in enumerate(self._model_spec.event_list):
            for trans in event.transition_list:
                if trans.transition_type==TransitionType.T:         # NOTE: are we ignoring people who are born infected?
                    orig = trans.origin
                    dest = trans.destination

                    if (orig not in infected_state_ids) and (dest in infected_state_ids):
                        idx = infected_state_ids.index(dest)
                        F[idx] += event_rate_vector[event_i]

        self._sympy_expression = F

class DiseaseProgressionVector(NumericMethod):
    method_name = "disease_progression_vector"
    depends_on = [
        "ode",
        "infection_vector"
    ]

    def _build_expression(self):

        infected_state_ids = self._model_spec.get_states_by_tag("infected")

        infected_indices = [self._model_spec.get_state_index(s) for s in infected_state_ids]

        ode = self._method_register["ode"].get_equation()

        disease_ode = ode[infected_indices, :]

        F = self._method_register["infection_vector"].get_equation()

        self._sympy_expression = F - disease_ode

class InfectionMatrix(NumericMethod):
    method_name = "infection_matrix"
    depends_on = ["infection_vector"]

    def _build_expression(self):

        infected_state_ids = self._model_spec.get_states_by_tag("infected")

        infected_symbols = [
            self._model_spec.state_dict[s] for s in infected_state_ids
        ]

        F = self._method_register["infection_vector"].get_equation()

        self._sympy_expression = F.jacobian(infected_symbols)

class DiseaseProgressionMatrix(NumericMethod):
    method_name = "disease_progression_matrix"
    depends_on = ["disease_progression_vector"]

    def _build_expression(self):

        infected_state_ids = (self._model_spec.get_states_by_tag("infected"))

        infected_symbols = [
            self._model_spec.state_dict[s] for s in infected_state_ids
        ]

        V = self._method_register["disease_progression_vector"].get_equation()

        self._sympy_expression = V.jacobian(infected_symbols)

class R0(NumericMethod):
    method_name = "R0"
    depends_on = [
        "infection_matrix",
        "disease_progression_matrix",
        "disease_free_equilibrium"
    ]

    def _build_expression(self):

        F = self._method_register["infection_matrix"].get_equation()

        V = self._method_register["disease_progression_matrix"].get_equation()

        dfe = self._method_register["disease_free_equilibrium"].get_equation()

        if len(dfe)==0:
            self._sympy_expression = []

        else:
            dfe = dfe[0]
            F = F.subs(dfe)
            V = V.subs(dfe)

            K = F * V.inv()

            eigenvalues = [
                sympy.simplify(ev)
                for ev in K.eigenvals()
                if ev != 0
            ]

            self._sympy_expression = eigenvalues[0]         # NOTE: Assuming only 1 value of R0
