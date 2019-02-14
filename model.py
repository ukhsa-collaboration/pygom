'''
Created on 5 Feb 2019

@author: thomas.finnie
'''
import uuid

from schematics.models import Model
from schematics.types.compound import ModelType, ListType
from schematics.types import (StringType,
                              FloatType,
                              UUIDType,
                              ModelType,
                              IntType
                              )
import pygom.model

# Some base types
class StateType(StringType):
    '''
    Our own version of string type for states (in case we want to change later)
    '''

class StateListType(ListType):
    '''
    A list of states
    '''
    def __init__(self, **kwargs):
        kwargs['field'] = StateType
        super().__init__(**kwargs)


class ParameterType(StringType):
    '''
    Our own version of string type for parameters
    '''

class ParameterListType(ListType):
    '''
    A list of Parameters
    '''
    def __init__(self, **kwargs):
        kwargs['field'] = ParameterType
        super().__init__(**kwargs)


class ParameterValue(Model):
    '''
    A parameter and its value
    '''
    parameter = ParameterType(required=True)
    parameter_value = FloatType(required=True)

class ParameterValueList(Model):
    '''
    A list of parameters and their values
    '''
    parameters = ListType(ModelType(ParameterValue))

# Transitions
class TransitionTypeType(StringType):
    '''
    An enum of the possible types a transition can take
    '''
    choices = [x.name for x in pygom.model.transition.TransitionType]
    default = 'ODE'

class TransitionModel(Model):
    '''
    A representation of a :class:`pygom.model.Transition`
    '''
    origin = StateType(required=True)
    equation = StringType(required=True)
    destination = StateType()
    transition_type = TransitionTypeType()
    ID = UUIDType(default=uuid.uuid4)
    name = StringType()

    def instantiate(self):
        '''
        Take this representation and make a PyGOM transform from it
        Returns
        -------
        A `pygom.models.Transition` based on the values of self
        '''
        return pygom.model.Transition(**self.to_native())

class TransitionListType(ListType):
    '''
    A list of Parameters
    '''
    def __init__(self, **kwargs):
        kwargs['field'] = ModelType(TransitionModel)
        super().__init__(**kwargs)

# Ode models
class BaseOdeModel(Model):
    '''
    A representation of a :class:`pygom.model.BaseOdeModel`
    '''
    state = StateListType()
    param = ParameterListType()
#    derived_param
    transition = TransitionListType()
    birth_death = TransitionListType()
    ode = TransitionListType()

class DeterministicOdeModel(BaseOdeModel):
    '''
    A representation of a :class:`pygom.model.DeterministicOde`
    '''
    model_type = StringType(default='DeterministicOde')

    def instantiate(self):
        '''
        Take this representation and make a PyGOM transform from it
        Returns
        -------
        A `pygom.models.DeterministicOde` based on the values of self
        '''
        #get our current state
        state = self.to_native()
        #remove the model type (needed for disabugation)
        state.pop('model_type')
        #convert transitions
        for t_list in ['transition', 'birth_death', 'ode']:
            state[t_list] = [TransitionModel(x).instantiate() for x in state[t_list]]
        #build the model
        return pygom.model.DeterministicOde(**state)

