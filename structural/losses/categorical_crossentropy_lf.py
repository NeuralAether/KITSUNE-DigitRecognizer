"""
Categorical crossentropy Loss Function definition 
"""

from config import * 
from structural.loss_function import LossFunctionJax
from structural.model import JaxModel

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

    def compute_grad(self, model: JaxModel, batch: jnp.ndarray, truth: jnp.ndarray) : 
        """
        To compute the gradients of the loss function with respect to the model parameters 
        """
        def is_top_leaf(x): 
            if not isinstance(x, dict):
                return True
            if any(k in x for k in ("kernel", "bias", "b")):
                return True
            if len(x) == 0:  # empty dict = leaf
                return True
            return False
        original_weights = model.get_weights()
        pure_weights = jax.tree_util.tree_map(lambda x: x["weights"], original_weights, is_leaf=lambda x: not isinstance(x, dict) or "weights" in x or "name" in x or "id" in x)
        weight_headers = jax.tree_util.tree_map(lambda x: {"name": x["name"], "id": x["id"]}, original_weights, is_leaf=lambda x: isinstance(x, dict) and "weights" in x)
        def loss_with_pure_weights(pure_weights, weight_headers, batch, truth): 
            # Reconstruct weights with headers
            reconstructed_weights = jax.tree_util.tree_map(
                lambda w, wh: {"weights": w, "name": wh["name"], "id": wh["id"]},
                pure_weights,
                weight_headers,
                is_leaf=is_top_leaf
            )
            copy_model = model.copy()
            copy_model.set_weights(reconstructed_weights)
            __logits = copy_model.forward(batch)
            loss_value = self.compute(__logits, truth)
            del copy_model
            return loss_value
        grads = jax.grad(lambda x: loss_with_pure_weights(x, weight_headers, batch, truth))(pure_weights)
        return grads, pure_weights, weight_headers