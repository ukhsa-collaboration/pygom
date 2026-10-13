"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module containing functions that performs epidemiology based analysis
    via algebraic manipulation, such as the basic reproduction number

"""

import sympy

# from .simulate import SimulateOde, TransitionType

# __all__ = [
#            'DFE',
#            'R0',
#            'R0_from_matrix',
#            'disease_progression_matrices'
#            ]

# These could be math methods:

from .maths.mathsmethod import NumericMethod

from sympy import Matrix

class Equilibrium(NumericMethod):

    method_name = "equilibrium"

    depends_on = ["ode"]

    def get_equation(self):
        
        dynamic_states = self._model_spec._state_store.get_dynamic_states()

        indices = [
            self._model_spec.get_state_index(str(state))        # TODO: this is not safe to assume str(state) comes from symbol
            for state in dynamic_states
        ]

        ode = self._method_register["ode"].get_equation()
        ode = ode[indices, :]

        constraints = Matrix([
            symb - expr
            for symb, expr in self._model_spec._derived_parameter_store.symbol_to_expression_dict.items()
        ])

        eqns = Matrix.vstack(
            ode,
            constraints
        )

        return sympy.solve(
            eqns,
            dynamic_states,
            dict=True
        )

class DiseaseFreeEquilibrium(NumericMethod):

    method_name = "disease_free_equilibrium"

    depends_on = ["equilibrium"]

    def get_equation(self):

        infected_states = self._model_spec._state_store.get_states_by_tag("infected")

        equilibria = self._method_register["equilibrium"].get_equation()

        return [
            eq
            for eq in equilibria
            if all(
                sympy.simplify(
                    eq.get(state, 0)
                ) == 0
                for state in infected_states
            )
        ]


class DiseaseFreeEquilibrium(NumericMethod):

    method_name = "disease_free_equilibrium"

    depends_on = ["equilibrium"]

    def get_equation(self):

        infected_states = self._model_spec._state_store.get_states_by_tag("infected")

        equilibria = self._method_register["equilibrium"].get_equation()

        return [
            eq
            for eq in equilibria
            if all(
                sympy.simplify(
                    eq.get(state, 0)
                ) == 0
                for state in infected_states
            )
        ]


# def DFE(ode, disease_state):
#     '''
#     Returns the disease free equilibrium from an ode object

#     Parameters
#     ----------
#     ode: :class:`.BaseOdeModel`
#         a class object from pygom
#     diseaseState: array like
#         name of the disease states

#     Returns
#     -------
#     e: array like
#         disease free equilibrium

#     '''

#     eqn = ode.ode.get_equation()
#     index = ode.get_state_index(disease_state)
#     states = [s for s in ode._iterStateList()]
#     states_subs = {states[i]: 0 for i in index}
#     eqn = eqn.subs(states_subs)

#     DFE_solution = sympy.solve(eqn, states)
#     if len(DFE_solution) == 0: DFE_solution = {}

#     for s in states:
#         if s not in states_subs.keys() and s not in DFE_solution.keys():
#             DFE_solution.setdefault(s, 0)
#     return DFE_solution




# from .model_spec import ModelSpec

from .transition import TransitionType

class InfectionVector(NumericMethod):

    method_name = "infection_vector"

    depends_on = [
        "event_rate_vector",
        "state_change_matrix"
    ]

    """
    Rates at which new infections occur
    """

    def get_equation(self):

        infected_states = self._model_spec._state_store.get_states_by_tag("infected")

        event_rate_vector = self._method_register["event_rate_vector"].get_equation()

        # container for output:
        F = sympy.zeros(len(infected_states), 1)

        for event_i, event in enumerate(self._model_spec.event_list):
            for trans in event.transition_list:
                if trans.transition_type==TransitionType.T:
                    orig = trans.origin
                    dest = trans.destination

                    print(dest)
                    print(type(dest))
                    print(sympy.Symbol(dest))
                    print(type(sympy.Symbol(dest)))
                    print(infected_states)
                    print(type(infected_states[0]))


                    print((sympy.Symbol(dest) in infected_states))
                    print(dest in infected_states)

                    if (sympy.Symbol(orig) not in infected_states) and (sympy.Symbol(dest) in infected_states):
                        print("yes")
                        idx = infected_states.index(dest)
                        F[idx] += event_rate_vector[event_i]

        return F


















# def R0(ode, disease_state):
#     '''
#     Returns the basic reproduction number, in symbolic form when
#     the parameter values are not available

#     Parameters
#     ----------
#     ode: :class:`.BaseOdeModel`
#         a class object from pygom
#     diseaseStateIndex: array like
#         name of the disease states

#     Returns
#     -------
#     e: array like
#         R0

#     See Also
#     --------
#     :func:`getDiseaseProgressionMatrices`, :func:`getR0GivenMatrix`

#     '''

#     F, V = disease_progression_matrices(ode, disease_state)
#     ## index = ode.get_state_index(disease_state)
#     e = R0_from_matrix(F, V)
#     DFE_eqn = DFE(ode, disease_state)
#     e = [eig.subs(DFE_eqn) for eig in e]
#     if ode.parameters is not None:
#         e = [eig.subs(ode.parameters) for eig in e]

#     e = list(filter(lambda x: sympy.Integer(-1) not in x.args, e))
#     return (e if len(e) > 1 else e[0])

# def R0_from_matrix(F, V, disease_state=None):
#     '''
#     Returns the symbolic form of the basic reproduction number. This will
#     include the states symbols which is different from :func:`getR0` where
#     the states is replaced by the values of the disease-free equilibrium.

#     Parameters
#     ----------
#     F: :class:`sympy.matrices.MatrixBase`
#         secondary infection rates
#     V: :class:`sympy.matrices.MatrixBase`
#         disease progression rates
#     disease_state: list like, optional
#         list of the disease state as :class:`sympy.Symbol`.  Defaults
#         to None which assumes that :math:`F,V` had been differentiated

#     Returns
#     -------
#     e: :class:`sympy.matrices.MatrixBase`
#         the eigenvalues of :math:`FV^{-1}` for the disease states

#     See Also
#     --------
#     :func:`getDiseaseProgressionMatrices`, :func:`getR0`
#     '''

#     if disease_state is None:
#         dF = F
#         dV = V
#     else:
#         dF = F.jacobian(disease_state)
#         dV = F.jacobian(disease_state)

#     K = dF*dV.inv()
#     e = K.eigenvals().keys()
#     e = filter(lambda x: x != 0, e)
#     return list(e)

# def disease_progression_matrices(ode, disease_states, diff=True):
#     '''
#     Returns (F,V), the secondary infection rates and disease progression
#     rate respectively.

#     Parameters
#     ----------
#     ode: :class:`.BaseOdeModel`
#         an ode class in pygom
#     disease_states: array like
#         the name of the disease states
#     diff: bool, optional
#         if the first derivative of the matrices are return, defaults to true

#     Returns
#     -------
#     (F, V): tuple
#         The progression matrices.  If diff=False, then we return the
#         :math:`F_{i}` and :math:`V_{i}` matrices as per [Brauer2008]_.
#     '''

#     #get the indicies of the disease states
#     diseaseIndex = [ode.get_state_index(disease_state)
#                     for disease_state in disease_states]
    
#     state_list = list()
#     for i, s in enumerate(ode._iterStateList()):
#         if i in diseaseIndex:
#             state_list.append(s)

#     FList = list()

#     for event in ode.event_list:
#         for trans in event.transition_list:
#             if trans.transition_type==TransitionType.T:
#                 orig = _get_single_state_name(trans.origin)
#                 dest = _get_single_state_name(trans.destination)
#                 if isinstance(orig, str) and isinstance(dest, str):
#                     if orig not in disease_states and dest in disease_states:
#                         trans._equation=event.rate
#                         FList.append(trans)

#     infected_submodel = SimulateOde(ode.state_list, ode.param_list, transition=FList)

#     F = ode2.ode.get_equation().row(diseaseIndex)
#     V = F - ode.ode.get_equation().row(diseaseIndex)

#     if diff:
#         dF = F.jacobian(state_list)
#         dV = V.jacobian(state_list)
#         return dF, dV
#     else:
#         return F,V


# def _get_single_state_name(state):
#     if hasattr(state, '__iter__'):
#         if isinstance(state, str):
#             return state
#         else:
#             state = state[0] if len(state) == 1 else None
#     if isinstance(state, str):
#         return state
#     elif isinstance(state, sympy.Symbol):
#         return str(state)
#     else:
#         return None
