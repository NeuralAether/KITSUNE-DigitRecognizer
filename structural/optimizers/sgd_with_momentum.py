"""
Gives a quick implementation of SGD with momentum optimizer.
"""

from structural.optimizer import OptimizerJax
from config import *

class SGDWithMomentumJax(OptimizerJax) :
    def __init__(self, learning_rate = 0.01, momentum = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.__velocities = None 
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self, grads, pure_weights, weight_headers, learning_rate=0.01, momentum=0.9, **kwargs): 
        """
        Performs a step of SGD with momentum 
        """
        def is_top_leaf(x): 
            if not isinstance(x, dict):
                return True
            if any(k in x for k in ("kernel", "bias", "b")):
                return True
            if len(x) == 0:  # empty dict = leaf
                return True
            return False

        if self.__velocities is None : 
            self.__velocities = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pure_weights)

        
        new_velocities = jax.tree_util.tree_map(
            lambda v, g: momentum * v - learning_rate * g,
            self.__velocities,
            grads
        )

        new_pure_weights = jax.tree_util.tree_map(
            lambda w, v: w + v,
            pure_weights,
            new_velocities
        )

        self.__velocities = new_velocities
        # Reconstruct weights with headers
        new_weights = jax.tree_util.tree_map(
            lambda w, wh: {"weights": w, "name": wh["name"], "id": wh["id"]},
            new_pure_weights,
            weight_headers, 
            is_leaf=is_top_leaf
        )

        return new_weights
    