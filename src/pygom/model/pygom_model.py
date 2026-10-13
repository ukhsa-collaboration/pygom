from .model_spec import ModelSpec

from .ode_utils.compiler import CompileCode

from .ode_utils.maths_methods_store import MathsMethodsRegistry

class DependencyManager:
    """
    Individual stores take care of their own caches for the most part,
    but sometimes dependency across different stores can lead to
    cache invalidation.

    Currently this deals with the simple case: adding new events invalidates
    the maths methods store. For now, the dependency manager is overly
    complicated, but ready if required for more complex model inter-
    dependency in the future.
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

        ##.....Simulation interface.....##

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


    # TODO:
    # 1) Transfer remaining mathmethods from DeterminsiticODE
    # 2) random number seeds