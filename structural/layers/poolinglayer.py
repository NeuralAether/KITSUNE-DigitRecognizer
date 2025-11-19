"""
The Pooling Layer in jax
"""

from config import * 
from structural.layer import JaxLayer

class PoolingJaxLayer(JaxLayer) : 
    """
    Implementation of a jax pooling layer
    """

    def __init__(self, type_= "avg",
                       window = (2,2), 
                       stride = (2,2), 
                       padding = "VALID",
                       **kwargs) : 
        super().__init__(**kwargs)
        self.__type = type_
        if isinstance(window, int) : 
            window = (window, window)
        if isinstance(stride, int) : 
            stride = (stride, stride)
        self.__window = window
        self.__stride = stride
        self.__padding = padding 
        self.__computation = None
        assert type_ in ["avg","sum","max"], "Invalid type"
        if self.__type == "max" : 
            self.__computation = jax.lax.max 
        else : 
            self.__computation = jax.lax.add

    def construct(self, batch, **kwargs):
        # Non trainable so no weight 
        try : 
            self.shapes = self(batch).shape
            self.shapes = (
                None, 
                self.shapes[1],
                self.shapes[2], 
                self.shapes[3]
            )
        except : 
            raise Exception("Could not correctly construct the model")
        
    def __call__(self, batch: jnp.ndarray) : 
        pooled_output = jax.lax.reduce_window(
            operand = batch, 
            init_value= 0.0, 
            computation=self.__computation,
            window_dimensions=(1,self.__window[0],self.__window[1],1),
            window_strides=  (1,self.__window[0],self.__window[1],1),
            padding=self.__padding
        )
        if self.__type == "avg" : 
            pooled_output = pooled_output / (self.__window[0]*self.__window[1])
        return pooled_output
    
    def get_summary(self):
        super().get_summary()
        return {"weights":{ 
        "trainable" : 0, 
            "shape" : self.shapes, 
        }, "name": self.name, "id" : self.id}