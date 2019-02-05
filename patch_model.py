'''
Created on 5 Feb 2019

@author: thomas.finnie
'''
from schematics.models import Model
from schematics.types.compound import ModelType, ListType

from .model import DeterministicOdeModel
from .movement import Flow
from .patch import Patch

class PatchModel(Model):
    patches = ListType(ModelType(Patch))
    flows = ListType(ModelType(Flow))
    template_model = ModelType(DeterministicOdeModel)
