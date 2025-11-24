"""
The structure of a layer in jax : To make it easier to code 
"""

from config import * 
import random, string

class JaxLayer : 
    """
    Implementation of a layer 
    """

    def __init__(self, **kwargs) : 
        # ----------- For all ---------------------
        self.id = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        self.name = kwargs.get("name","")
        # The random key : 
        self.layer_key = jax.random.PRNGKey(random.randint(0,1000))
        # Some weights
        self.weights = {}
        self.shapes = {}

    def construct(self, batch: jnp.ndarray , **kwargs) : 
        """
        Construct the layer
        """
        pass
        
    def __call__(self, batch: jnp.ndarray):
        """
        Code the call function in here 
        """
        # Return the call to that specific layer (using the weights)
        return batch
    
    """
    Some getter and setters 
    """

    def copy(self) : 
        """
        To copy the layer 
        """
        __new_layer = JaxLayer(name= self.name)
        __new_layer.weights = self.weights.copy()
        __new_layer.shapes = self.shapes # Copy shapes as well tuple
        __new_layer.layer_key = self.layer_key
        __new_layer.id = self.id + "_copy"
        __new_layer.name = self.name
        return __new_layer

    def get_weights(self): 
        return {"weights": self.weights, "name": self.name,"id" : self.id}
    
    def set_weights(self, weights): 
        self.weights = weights

    """
    The summary of the layer
    """

    def get_summary(self) : 
        if (self.weights == {} and len(self.shapes) == 0 ) :
            raise Exception("Model not constructed yet")