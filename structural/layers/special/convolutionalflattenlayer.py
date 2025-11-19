"""
The convolutional and flatten Layer (special for LeNet)
"""

from config import * 
from structural.layer import JaxLayer

class ConvolutionalAndFlattenJaxLayer(JaxLayer): 
    """
    Implementation of a jax Lenet convolutional and flatten layer
    Assumes square kernels and square images 
    """
    def __init__(self, out_channels, stride= None, 
                       padding= "VALID",
                       dimensions = {
                            "input": "NHWC", # Means Batch (N), Height, Width, and Channels
                            "kernel": "OIHW", # Means Out channels, Input Channels, Height, Width
                            "output": "NHWC"
                       } , **kwargs):
        """
        This applies a kernel but special kernel so that it flattens the images
        """
        super().__init__(**kwargs)
        if isinstance(stride, int) : 
            stride = (stride, stride)
        self.__out_channels = out_channels
        self.__stride = stride 
        self.__padding = padding 
        self.__dimensions = dimensions
        assert isinstance(stride, tuple), "Stride type not understood"

    def __get_size(self, length, stride) : 
        lower_bound = max(length - stride +1,1) 
        return lower_bound

    def construct(self, batch: jnp.ndarray , **kwargs) : 
        __kernel_size = (self.__get_size(batch.shape[1], self.__stride[0]),
                    self.__get_size(batch.shape[2], self.__stride[1]))
        

        print(f"Using the kernel size {__kernel_size}")
        kernel_key , self.layer_key =  jax.random.split(self.layer_key)
        # Weight construction 
        self.weights = {
            "kernel" : jax.random.normal(
                kernel_key, shape = (
                    self.__out_channels, # The amount of channels (and so amount of kernels)
                    batch.shape[-1], 
                    __kernel_size[0], # Height 
                    __kernel_size[1], # Width
                )
            ),
            "bias" : jax.random.normal(
                kernel_key, shape = (
                    self.__out_channels,
                )
            )
        }
        # The shape now (Very special in this case)
        try : 
            self.shapes = self(batch).shape
            self.shapes = (
                None, 
                self.shapes[1]
            )
        except : 
            raise Exception("Could not correctly construct the model")
        
    def __call__(self, batch):
        # The call to the layer : 
        convoluted_output = jax.lax.conv_general_dilated(
            lhs = batch, 
            rhs = self.weights["kernel"], 
            window_strides= self.__stride, 
            padding = self.__padding, 
            dimension_numbers= tuple(self.__dimensions.values())
        )
        convoluted_output += self.weights["bias"]
        if convoluted_output.shape[1] != 1 or convoluted_output.shape[2] != 1 : 
            print(convoluted_output.shape)
            raise Exception("Kernel is not flattening the image correctly")
        convoluted_output = convoluted_output.reshape((convoluted_output.shape[0],-1))
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