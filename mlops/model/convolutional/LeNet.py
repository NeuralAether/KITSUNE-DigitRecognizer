"""
Hommage to the first CNN to actually be successful (by Yann LeCunn 1998)

Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
keywords: {Neural networks;Pattern recognition;Machine learning;Optical character recognition software;Character recognition;Feature extraction;Multi-layer neural network;Optical computing;Hidden Markov models;Principal component analysis},
Page 2283 for figure  

The LeNet structure is : (technically for 32,32 images but will be adapted to 28,28 for our case)
Input : (None, 28, 28)
C1 : Convolutional -> Kernel (5,5) + Feature Maps 6, and stride 1 -> (None, 24, 24, 6)
S2 : Average Pooling (Subsampling) -> 2,2 kernel , stride 2 -> (None, 12, 12, 6) 
C3 : Convolutional -> Kernel (5,5) + 16 Feature Maps , stride 1 -> (None, 8,8 , 16)
S4 : Average Pooling  -> 2,2 kernel , stride 2 -> (None, 4,4,16)
C5 : Convolutional -> Kernel (4,4) -> To result in 1 map + 120 feature maps, and stride -> (None, 1,1, 120) (Originally 5,5 in the paper)
F6 : Fully Connected Layer -> 84 neurons (Activation tanh) -> (None, 84)
Output : A softmax function (we will be using linear in our case + logits) -> (None, 10)
"""

from config import * 

class LeNet : 
    """
    Implementation of the LeNet convolutional layer
    """
    def __init__(self):
        # Full structure
        self.__config = {
            "C1" : {"kernels" : {"Out Channels" : 6, "kh" : 5, "kw" : 5},
                            "stride" : {"sh" : 1, "sw" : 1}}, 
            "S2" : {"windows" : {"wh" : 2, "ww" : 2},
                    "stride" : {"sh" : 2,"sw" : 2}},
            "C3" : {"kernels" : {"Out Channels" : 16, "kh" : 5, "kw" : 5},
                    "stride" : {"sh" : 1, "sw" : 1}}, 
            "S4" : {"windows" : {"wh" : 2, "ww" : 2},
                    "stride" : {"sh" : 2,"sw" : 2}},
            "C5" : {"kernels" : {"Out Channels" : 120, "kh" : 4, "kw" : 4},
                    "stride" : {"sh" : 1, "sw" : 1}}, 
            "F6" : {"size" : 84},
            "Output" : {"size" : 10}
        }
        # Initializer keys
        self.__kernel_key = jax.random.PRNGKey(42)
        self.__weight_key = jax.random.PRNGKey(123)
        self.__weights = {}

    def construct(self, batch: jnp.ndarray) : 
        """
        Has to be run to 
        """
        if self.__weights != {} : 
            print("Not reconstructing to avoid losing the weights")
            return 
        # C1 : 
        self.__weights["C1"] = {
            "kernel_shape" : (self.__config["C1"]["kernels"]["Out Channels"], 
                        batch.shape[-1], 
                        self.__config["C1"]["kernels"]["kh"],
                        self.__config["C1"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C1"]["stride"]["sh"],
                        self.__config["C1"]["stride"]["sw"]),
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C1"]["kernels"]["Out Channels"], 
                        batch.shape[-1], 
                        self.__config["C1"]["kernels"]["kh"],
                        self.__config["C1"]["kernels"]["kw"]) ) 
        }
        # S2 : 
        self.__weights["S2"] = {
            "window_shape" : (self.__config["S2"]["windows"]["wh"],
                             self.__config["S2"]["windows"]["ww"]),
            "stride_shape" : (self.__config["S2"]["stride"]["sh"],
                        self.__config["S2"]["stride"]["sw"])
        }
        # C3 : 
        self.__weights["C3"] = {
            "kernel_shape" : (self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C1"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["kh"],
                        self.__config["C3"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C3"]["stride"]["sh"],
                        self.__config["C3"]["stride"]["sw"]),
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C1"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["kh"],
                        self.__config["C3"]["kernels"]["kw"]) ) 
        }
        # S4 : 
        self.__weights["S4"] = {
            "window_shape" : (self.__config["S4"]["windows"]["wh"],
                             self.__config["S4"]["windows"]["ww"]),
            "stride_shape" : (self.__config["S4"]["stride"]["sh"],
                        self.__config["S4"]["stride"]["sw"])
        }
        # C5 : 
        self.__weights["C5"] = {
            "kernel_shape" : (self.__config["C5"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C5"]["kernels"]["kh"],
                        self.__config["C5"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C5"]["stride"]["sh"],
                        self.__config["C5"]["stride"]["sw"]),
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C5"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C5"]["kernels"]["kh"],
                        self.__config["C5"]["kernels"]["kw"]) ) 
        }
        # F6 : 
        self.__weights["F6"] = {
            "W" : jax.random.normal(self.__weight_key, shape=(self.__config["C5"]["kernels"]["Out Channels"], self.__config["F6"]["size"])) , 
            "b" : jax.random.normal(self.__weight_key, shape=(self.__config["F6"]["size"], )), 
        }
        # Output : 
        self.__weights["Output"] = {
            "W" : jax.random.normal(self.__weight_key, shape=(self.__config["F6"]["size"], self.__config["Output"]["size"])) , 
            "b" : jax.random.normal(self.__weight_key, shape=(self.__config["Output"]["size"], )), 
        }

    """
    The forward functions 
    """

    def forward(self, batch: jnp.ndarray): 
        """
        The forward pass: Will actually return the logits
        """
        if self.__weights == {} : 
            # We compile the model first 
            print("Constructing the model first ...")
            self.construct(batch)
        
        # C1 : 
        C1 = self.__convolutional_layer(batch, self.__weights["C1"]["kernel"], stride = self.__weights["C1"]["stride_shape"])
        # S2 :
        S2 = self.__pooling_layer(C1, window= self.__weights["S2"]["window_shape"], stride= self.__weights["S2"]["stride_shape"] )
        # C3 : 
        C3 = self.__convolutional_layer(S2, self.__weights["C3"]["kernel"], stride = self.__weights["C3"]["stride_shape"])
        # S4 : 
        S4 = self.__pooling_layer(C3, window= self.__weights["S4"]["window_shape"], stride= self.__weights["S4"]["stride_shape"]  )
        # C5 : 
        C5 = self.__convolutional_layer(S4, self.__weights["C5"]["kernel"], stride = self.__weights["C5"]["stride_shape"])
        C5 = C5.reshape((batch.shape[0], -1)) # Full flat now 
        # F6 : 
        F6 = self.__fully_connected_layer(C5, self.__weights["F6"]["W"], self.__weights["F6"]["b"] )
        # Output : 
        logits = self.__fully_connected_layer(F6, self.__weights["Output"]["W"], self.__weights["Output"]["b"] , activation=None)
        return logits
    
    def predict(self, batch: jnp.ndarray) : 
        """
        This predicts and uses the logits to get the softmax layer , not used at training
        """
        logits = self.forward(batch)
        max_logit = jnp.max(logits, axis= 1).reshape(-1,1)
        print(max_logit)
        corrected_logits = logits - max_logit 
        exp_logits = jnp.exp(corrected_logits)
        sum_exp_logits = jnp.sum(exp_logits, axis=1).reshape(-1,1)
        probabilities = exp_logits/sum_exp_logits
        return probabilities, jnp.argmax(probabilities, axis=1) +1
    
    """
    The backward functions
    """

    def compile(self) : 
        pass

    """
    The layers
    """
    def __convolutional_layer(self, batch: jnp.ndarray, 
                                    kernel: jnp.ndarray, 
                                    stride = (1,1), # In both width and height
                                    padding = "VALID", 
                                    dimensions = {
                                        "input": "NHWC", # Means Batch (N), Height, Width, and Channels
                                        "kernel": "OIHW", # Means Out channels, Input Channels, Height, Width
                                        "output": "NHWC"
                                    }, ) : 
        """
        Convolutional Layer 
        """
        # Will be checked later
        convolved_output = jax.lax.conv_general_dilated(
            lhs = batch, 
            rhs = kernel, 
            window_strides=stride, 
            padding = padding, 
            dimension_numbers=tuple(dimensions.values())
        )
        return convolved_output

    def __pooling_layer(self, batch: jnp.ndarray,
                              type = "avg", 
                              computation = jax.lax.add, # Average pooling is actually like add 
                              window = (2,2), 
                              stride = (2,2), 
                              padding = "VALID"
                        ) : 
        assert type in ["avg", "sum", "max"], "Invalid type"
        if type == "max" : 
            assert computation == jax.lax.max, "Invalid function"
        pooled_output = jax.lax.reduce_window(
            operand= batch, 
            init_value= 0.0, # Always take 0 
            computation= computation, 
            window_dimensions= (1,window[0],window[1],1), 
            window_strides= (1,stride[0],stride[1],1),
            padding=padding
        )
        if type== "avg" :
            pooled_output = pooled_output / (window[0]*window[1]) 
        return pooled_output
    
    def __fully_connected_layer(self, batch: jnp.ndarray,
                                      W : jnp.ndarray, 
                                      b : jnp.ndarray, 
                                      activation = jnp.tanh) -> jnp.ndarray : 
        """
        A fully connected layer
        """
        output = jnp.dot(batch, W) + b # g(wx+b)
        if activation : 
            output = activation(output)
        return output 