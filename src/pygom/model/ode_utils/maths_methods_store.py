"""
Store maths methods
"""

from ..maths.state_change_matrix import StateChangeMatrix
from ..maths.ode_system import ODESystem
from ..maths.event_rate_vector import EventRateVector
from ..maths.jacobian import ODEJacobianParams, ODEJacobianStates, EventRatesJacobianParams, EventRatesJacobianStates
from ..maths.hessian import ODEHessianParams, ODEHessianStates, EventRatesHessianParams, EventRatesHessianStates
from ..maths.hessian_mixed import ODEMixedHessianStatesParams, EventRatesMixedHessianStatesParams

from ..epi_analysis import Equilibrium, DiseaseFreeEquilibrium, InfectionVector


class MathsMethodsRegistry:

    _maths_methods = [
        # Fundamental objects
        EventRateVector,
        ODESystem,
        EventRateVector,
        StateChangeMatrix,

        # Jacobians of the above
        ODEJacobianParams,
        ODEJacobianStates,
        EventRatesJacobianParams,
        EventRatesJacobianStates,

        # Hessians of the above
        ## d/dx^2 or d/dp^2 (wrt states or params)
        ODEHessianParams,
        ODEHessianStates,
        EventRatesHessianParams,
        EventRatesHessianStates,
        ## d/dxdp   (mixed wrt states and params)
        ODEMixedHessianStatesParams,
        EventRatesMixedHessianStatesParams,

        # Epi modules
        Equilibrium,
        DiseaseFreeEquilibrium,
        InfectionVector

    ]

    def __init__(
            self,
            model_spec,
            compiler,
            dependency_manager
        ):
        self._dependency_manager = dependency_manager
        self._model_spec = model_spec
        self._SC = compiler

        self._method_classes = dict()
        self._methods = dict()

        # Populate the registry with all the methods
        for method in self._maths_methods:
            name = method.method_name
            self._register(method)
            setattr(
                self,
                name,
                self[name]
            )

    def __getitem__(self, name):
        if self._dependency_manager.is_dirty("maths_methods"):
            self._invalidate_caches()
            self._dependency_manager.mark_clean("maths_methods")

        if name not in self._methods:
            self._build(name)

        return self._methods[name]

    def _register(self, method_cls):
        self._method_classes[method_cls.method_name] = method_cls

    def _build(self, name):
        cls = self._method_classes[name]

        for dependency in cls.depends_on:
            self._build(dependency)

        # Create an instance of the maths class with this class as the 
        # associated ode system
        maths_class_instance = cls(
            self._model_spec,
            self._SC,
            self
        )

        self._methods[name] = maths_class_instance

    def _invalidate_caches(self)->None:
        """
        Tell objects that have cached components to reset their caches as
        the underlying system has changed
        """
        # The maths methods
        for mathsmethod in self._maths_methods:
            try:
                method_instance = getattr(self, mathsmethod.method_name)
                method_instance.invalidate_cache()
            except AttributeError:
                pass # We may not yet have all the objects
        
        # # The states_and_parameters list (none === not set)
        # # NOTE: Are we saying "math methods have changed, thus invalidating states/params"
        # #       not sure if it applies in that direction.
        # self._sp = None