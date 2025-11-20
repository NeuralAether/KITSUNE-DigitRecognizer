"""
Defines a loss function class
"""

from config import * 

class LossFunctionJax : 
    def __init__(self, **kwargs):
        pass
        
    def compute(self, predicted, truth) : 
        pass

    def getgrad(self, predicted, truth, weights= None): 
        pass