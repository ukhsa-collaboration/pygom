from .model_spec import ModelSpec

from .maths.state_change_matrix import StateChangeMatrix
from .maths.ode_system import ODESystem
from .maths.event_rate_vector import EventRateVector

# from .maths.grad import ODEGradParams, ODEGradStates, EventRatesGradParams, EventRatesGradStates
from .maths.jacobian import ODEJacobianParams, ODEJacobianStates, EventRatesJacobianParams, EventRatesJacobianStates

from .maths.hessian import ODEHessianParams, ODEHessianStates, EventRatesHessianParams, EventRatesHessianStates

from .maths.hessian_mixed import ODEMixedHessianStatesParams, EventRatesMixedHessianStatesParams

# from .maths.transition_jacobian import TransitionJacobian
# from .maths.hessian import Hessian

from .ode_utils import compileCode



class MathsRegistry:

    def __init__(self, model_spec, compiler):
        self._model_spec = model_spec
        self._SC = compiler

        self._method_classes = dict()
        self._methods = dict()

    def register(self, method_cls):
        self._method_classes[method_cls.method_name] = method_cls

    def __getitem__(self, name):
        if name not in self._methods:
            self._build(name)

        return self._methods[name]

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
        
        # The states_and_parameters list (none === not set)
        # NOTE: Are we saying "math methods have changed, thus invalidating states/params"
        #       not sure if it applies in that direction.
        self._sp = None

class Model():

    _maths_methods = [
        # Fundamental objects
        StateChangeMatrix,
        ODESystem,
        EventRateVector,

        # Jacobians of the above
        ODEJacobianParams,
        ODEJacobianStates,
        EventRatesJacobianParams,
        EventRatesJacobianStates,

        # Hessians of the above
        ## d/dx^2 or d/dp^2
        ODEHessianParams,
        ODEHessianStates,
        EventRatesHessianParams,
        EventRatesHessianStates,
        ## d/dxdp
        ODEMixedHessianStatesParams,
        EventRatesMixedHessianStatesParams


    ]

    # TODO: maths methods need to include derivatives of derived params.

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            event=None,
            backend='lambda'
        ):

        ## Foundational model specifications ##
        # (states, params, events)
        self._ModelSpec = ModelSpec(
            state=state,
            param=param,
            derived_param=derived_param,
            event=event
        )

        # Compiler
        self._SC = compileCode(backend=backend)

        ## Maths methods registry ##
        self._MethodRegister = MathsRegistry(
            model_spec=self._ModelSpec,
            compiler=self._SC
        )

        for method in self._maths_methods:
            name = method.method_name
            self._MethodRegister.register(method)
            setattr(
                self,
                name,
                self._MethodRegister[name]
            )

        ##.....Simulation interface.....##

        ##.....Loss functions interface.....##

        ##.....ABC interface.....##


    @property
    def parameters(self):
        """

        """
        return self._ModelSpec.parameters

    @parameters.setter
    def parameters(self, parameters):
        self._ModelSpec.parameters = parameters


    # TODO:

    # 1) User accessible attributes
    # 2) Cache and setting params
    # 3) Transfer remaining mathmethods from DeterminsiticODE
    # 4) random number seeds