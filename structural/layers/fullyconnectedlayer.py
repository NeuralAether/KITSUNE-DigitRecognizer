"""
Very simple dense layer
"""

from config import * 
from structural.layer import JaxLayer

class FullyConnectedJaxLayer(JaxLayer) : 
    """
    Implementation of Fully Connected Layer
    """
    def __init__(self, out_channels, 
                       activation = lambda x: x , # No activation 
                       **kwargs): 
        super().__init__(**kwargs)
        self.__out_channels = out_channels
        self.__activation = activation
        assert isinstance(self.__out_channels,int), "Out channels should be integer"

    def construct(self, batch, **kwargs):
        assert len(batch.shape) == 2 , "Unknown batch size"
        in_channels = batch.shape[-1]
        weight_key , self.layer_key = jax.random.split(self.layer_key)
        # Updating the weights
        self.weights = {
            "W" : jax.random.normal(weight_key, shape=(in_channels,self.__out_channels)) , 
            "b" : jax.random.normal(weight_key, shape = (self.__out_channels))
        }
        try : 
            self.shapes = self(batch).shape
            self.shapes = (None, self.shapes[1])
        except : 
            raise Exception("Could not correctly construct the model")

    def __call__(self, batch):
        # Very simple output : 
        output = jnp.dot(batch, self.weights["W"]) + self.weights["b"]
        output = self.__activation(output)
        return output
    
    def get_summary(self):
        super().get_summary()
        trainable_weights = math.prod(self.weights["W"].shape) 
        trainable_bias = math.prod(self.weights["b"].shape)
        return {"weights":{ "trainable_details" : {
            "weights" : trainable_weights, 
            "bias" : trainable_bias
        }, "trainable" : trainable_weights + trainable_bias, 
            "shape" : self.shapes, 
        }, "name": self.name, "id" : self.id}
