"""
Categorical crossentropy Loss Function definition 
"""

from config import * 
from structural.loss_function import LossFunctionJax

class CCLossFunctionJax(LossFunctionJax) : 
    """
    Implementation of the categorical crossentropy 
    """
    def __init__(self, from_logits= False , **kwargs):
        super().__init__(**kwargs)
        self.__from_logits = from_logits

    def __compute_logprobabilities_from_logits(self, logits: jnp.ndarray): 
        """
        Having logits, computes the probabilities from them 
        """
        corrected_logits = logits - jnp.max(logits, axis=1, keepdims=True)
        log_probabilities = corrected_logits - jnp.log(jnp.sum(jnp.exp(corrected_logits), axis=-1, keepdims=True))
        return log_probabilities
    
    def compute(self, predicted: jnp.ndarray, truth: jnp.ndarray): 
        """
        Forward loss function 
        """
        if self.__from_logits : 
            __log_probabilities = self.__compute_logprobabilities_from_logits(predicted)
            mean_categorical_cross = jnp.mean(-__log_probabilities[jnp.arange(predicted.shape[0]), truth]) 
        else : 
            __log_probabilities = jnp.log(predicted) # Already are probabilities
            mean_categorical_cross = jnp.mean(-__log_probabilities[jnp.arange(predicted.shape[0]), truth]) 
        return mean_categorical_cross

    