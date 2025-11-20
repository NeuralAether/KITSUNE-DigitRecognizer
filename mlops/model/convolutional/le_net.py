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
from structural.model import JaxModel
from structural.layers.fully_connected_layer import FullyConnectedJaxLayer
from structural.layers.convolutional_layer import ConvolutionalJaxLayer
from structural.layers.pooling_layer import PoolingJaxLayer
from structural.layers.special.convolutional_flatten_layer import ConvolutionalAndFlattenJaxLayer

class LeNetJaxImplementation(JaxModel) : 
    """
    LeNet Jax Model using predefined layers. All layers are custom. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The structure 
        # ----- STATIC ----- 
        self.__c1 = ConvolutionalJaxLayer(6, kernel_size=5, stride=1, name="C1")
        self.__s2 = PoolingJaxLayer("avg", window=2, stride=2, name="S2")
        self.__c3 = ConvolutionalJaxLayer(16, kernel_size=5, stride=1, name="C3")
        self.__s4 = PoolingJaxLayer("avg", window=2, stride=2, name="S4") 
        self.__c5 = ConvolutionalAndFlattenJaxLayer(120, stride=1, name="C5") # The kernel size is automatic
        self.__f6 = FullyConnectedJaxLayer(84, activation=jnp.tanh, name="F6")
        self.__output = FullyConnectedJaxLayer(10, name="Output") # Made such that no activation is linear

        # ----- DYNAMIC (Weights) ----
        # The definition 
        self.layers = [
            self.__c1, 
            self.__s2,
            self.__c3, 
            self.__s4, 
            self.__c5, 
            self.__f6,
            self.__output
        ] 

    """
    The prediction function is interesting for this : 
    """

    def predict(self, batch: jnp.ndarray, **kwargs) : 
        __logits = self.forward(batch)
        __exp_logits = jnp.exp(__logits - jnp.max(__logits, axis= 1, keepdims=True) )
        __sum_exp_logits = jnp.sum(__exp_logits, axis= 1, keepdims=True)
        return {
            "probabilities" : __exp_logits/__sum_exp_logits, 
            "predicted_labels" : jnp.argmax(__exp_logits/__sum_exp_logits, axis=1)
        }
    
    