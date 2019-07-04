'''
Created on 4 Feb 2019

@author: thomas.finnie
'''
import uuid

from schematics.models import Model
from schematics.types import (FloatType,
                              UUIDType,
                              ListType,
                              ModelType,
                              IntType,
                              StringType,
                              PolyModelType
                              )

from .model import StateValue
from bokeh.io import state

class Flux(Model):
    '''
    A record of where population comes from and goes to in a psudo-patch
    Parameters
    ----------
    time_point: The time point for which this flux is valid
    patch_id: The patch to which this flux applies
    flow_id: The flow to which this flux belongs
    origin: The id of a `Patch` from where the population come
    destination: The id of a `Patch` where will this population go next
    '''
    time_point = IntType()
    patch_id = UUIDType()
    flow_id = UUIDType()
    origin_id = UUIDType()
    origin_time_point = IntType()
    destination_id = UUIDType()
    destination_time_point = IntType()
#    value = FloatType(min_value=0) # How big a flux?

class Appointment(Model):
    '''
    The place (patch) and 'time' for a `Flow`

    Parameters
    ----------
    patch: The id of a patch
    timepoint: The timepoint that this flow is in the patch
    '''
    patch = UUIDType()
    time_point = IntType()

# class FlowValueAbsolute(Model):
#     '''
#     A flow value model that contains the absolute numbers for each initial
#     state as state values
#     '''
#     absolute_values = ListType(ModelType(StateValue))
#
#     @classmethod
#     def _claim_polymorphic(cls, data):
#         return data.get('absolute_values') is not None
#
#
# class FlowValueProportional(Model):
#     '''
#     A flow value model that allows the specification of a single state value and
#     proportions that should be assigned to each state
#     '''
#     total_value = FloatType(min_value=0)
#     proportion_values = ListType(FloatType(min_value=0))
#     @classmethod
#     def _claim_polymorphic(cls, data):
#         return data.get('proportion_values') is not None

class Flow(Model):
    '''
    A Flow is the distinct travel pattern of a group of individuals

    Parameters
    ----------
    id: The id of the flow
    initial_state_values: The number of individuals in the flow in each state
      at the start of the model.
    circad: A list of times and places that a flow visits
    '''
    id = UUIDType(default=uuid.uuid4)
    initial_state_values = ListType(ModelType(StateValue))
    circad = ListType(ModelType(Appointment))
