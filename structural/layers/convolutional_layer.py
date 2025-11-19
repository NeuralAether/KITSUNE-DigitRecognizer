"""
The Convolutional Layer in jax
"""

from config import * 
from structural.layer import JaxLayer 

class ConvolutionalJaxLayer(JaxLayer) : 
    """
    Implementation of a jax convolutional layer
    """
    def __init__(self, out_channels , kernel_size = 5, 
                       stride = 1, 
                       padding = "VALID", 
                       dimensions = {
                                        "input": "NHWC", # Means Batch (N), Height, Width, and Channels
                                        "kernel": "OIHW", # Means Out channels, Input Channels, Height, Width
                                        "output": "NHWC"
                                    },
                       **kwargs):
        """
        outchannels and kernel_size have to be given (either an int , or a tuple )
        """
        super().__init__(**kwargs)
        self.__out_channels = out_channels
        if isinstance(kernel_size , int) : 
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride , int) : 
            stride = (stride, stride)
        self.__kernel_size = kernel_size
        self.__stride = stride 
        self.__padding = padding 
        self.__dimensions = dimensions 
        assert self.__kernel_size is not None and isinstance(self.__kernel_size,tuple), "At least one is given , if kernel is given it overwrites kernel_size"
        assert isinstance(stride, tuple) , "Stride type not understood"
        # Other checks to do 

    def construct(self, batch: jnp.ndarray, **kwargs) :
        kernel_key, self.layer_key = jax.random.split(self.layer_key)
        # Weight construction 
        self.weights = {
            "kernel" : jax.random.normal(
                kernel_key, shape = (
                    self.__out_channels, # The amount of channels (and so amount of kernels)
                    batch.shape[-1], 
                    self.__kernel_size[0], # Height 
                    self.__kernel_size[1], # Width
                )
            ),
            "bias" : jax.random.normal(
                kernel_key, shape = (
                    self.__out_channels,
                )
            )
        }
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

    def __call__(self, batch : jnp.ndarray):
        # The call to the layer : 
        convoluted_output = jax.lax.conv_general_dilated(
            lhs = batch, 
            rhs = self.weights["kernel"], 
            window_strides= self.__stride, 
            padding = self.__padding, 
            dimension_numbers= tuple(self.__dimensions.values())
        )
        convoluted_output += self.weights["bias"]
        
        return convoluted_output
    
    def get_summary(self):
        super().get_summary()
        trainable_weights_kernel = math.prod(self.weights["kernel"].shape) 
        trainable_weights_bias = math.prod(self.weights["bias"].shape)
        
        return {"weights":{ "trainable_details" : {
            "kernel" : trainable_weights_kernel, 
            "bias" : trainable_weights_bias
        }, "trainable" : trainable_weights_bias + trainable_weights_kernel, 
            "shape" : self.shapes, 
        }, "name": self.name, "id" : self.id}
