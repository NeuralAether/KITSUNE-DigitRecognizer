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
import pickle, os
from tqdm import tqdm

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
        self.__shapes = {}
        # The training variables 
        self.__velocities = {}
        # The shape variables
        self.__total_train_batches = 0
        self.__total_test_batches = 0

    def construct(self, batch: jnp.ndarray) : 
        """
        Has to be run to 
        """
        if self.__weights != {} : 
            print("Not reconstructing to avoid losing the weights")
            return 
        # C1 : 
        self.__weights["C1"] = {
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C1"]["kernels"]["Out Channels"], 
                        batch.shape[-1], 
                        self.__config["C1"]["kernels"]["kh"],
                        self.__config["C1"]["kernels"]["kw"]) ) 
        }
        self.__shapes["C1"] = {
            "kernel_shape" : (self.__config["C1"]["kernels"]["Out Channels"], 
                        batch.shape[-1], 
                        self.__config["C1"]["kernels"]["kh"],
                        self.__config["C1"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C1"]["stride"]["sh"],
                        self.__config["C1"]["stride"]["sw"]),
        }
        # S2 : (Non trainable)
        self.__shapes["S2"] = {
            "window_shape" : (self.__config["S2"]["windows"]["wh"],
                             self.__config["S2"]["windows"]["ww"]),
            "stride_shape" : (self.__config["S2"]["stride"]["sh"],
                        self.__config["S2"]["stride"]["sw"])
        }
        # C3 : 
        self.__weights["C3"] = {
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C1"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["kh"],
                        self.__config["C3"]["kernels"]["kw"]) ) 
        }
        self.__shapes["C3"] = {
            "kernel_shape" : (self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C1"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["kh"],
                        self.__config["C3"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C3"]["stride"]["sh"],
                        self.__config["C3"]["stride"]["sw"]),
        }
        # S4 : (None trainable)
        self.__shapes["S4"] = {
            "window_shape" : (self.__config["S4"]["windows"]["wh"],
                             self.__config["S4"]["windows"]["ww"]),
            "stride_shape" : (self.__config["S4"]["stride"]["sh"],
                        self.__config["S4"]["stride"]["sw"])
        }
        # C5 : 
        self.__weights["C5"] = {
            "kernel" : jax.random.normal(self.__kernel_key, shape= (self.__config["C5"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C5"]["kernels"]["kh"],
                        self.__config["C5"]["kernels"]["kw"]) ) 
        }
        self.__shapes["C5"] = {
            "kernel_shape" : (self.__config["C5"]["kernels"]["Out Channels"], 
                        self.__config["C3"]["kernels"]["Out Channels"], 
                        self.__config["C5"]["kernels"]["kh"],
                        self.__config["C5"]["kernels"]["kw"]), 
            "stride_shape" :  (self.__config["C5"]["stride"]["sh"],
                        self.__config["C5"]["stride"]["sw"]),}
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

    def forward(self, batch: jnp.ndarray, weights = None): 
        """
        The forward pass: Will actually return the logits
        """
        if self.__weights == {} : 
            # We compile the model first 
            print("Constructing the model first ...")
            self.construct(batch)
        if weights is None : 
            weights = self.__weights
    
        # C1 : 
        C1 = self.__convolutional_layer(batch, weights["C1"]["kernel"], stride = self.__shapes["C1"]["stride_shape"])
        # S2 :
        S2 = self.__pooling_layer(C1, window= self.__shapes["S2"]["window_shape"], stride= self.__shapes["S2"]["stride_shape"] )
        # C3 : 
        C3 = self.__convolutional_layer(S2, weights["C3"]["kernel"], stride = self.__shapes["C3"]["stride_shape"])
        # S4 : 
        S4 = self.__pooling_layer(C3, window= self.__shapes["S4"]["window_shape"], stride= self.__shapes["S4"]["stride_shape"]  )
        # C5 : 
        C5 = self.__convolutional_layer(S4, weights["C5"]["kernel"], stride = self.__shapes["C5"]["stride_shape"])
        C5 = C5.reshape((batch.shape[0], -1)) # Full flat now 
        # F6 : 
        F6 = self.__fully_connected_layer(C5, weights["F6"]["W"], weights["F6"]["b"], activation= jnp.tanh )
        # Output : 
        logits = self.__fully_connected_layer(F6, weights["Output"]["W"], weights["Output"]["b"] , activation=None)
        return logits
    
    def predict(self, batch: jnp.ndarray, weights= None) : 
        """
        This predicts and uses the logits to get the softmax layer , not used at training
        """
        logits = self.forward(batch, weights)
        max_logits = jnp.max(logits, axis= 1, keepdims=True)
        corrected_logits = logits - max_logits
        exp_logits = jnp.exp(corrected_logits)
        sum_exp_logits = jnp.sum(exp_logits, axis=1, keepdims=True)
        probabilities = exp_logits/sum_exp_logits
        return probabilities, jnp.argmax(probabilities, axis=1)
    
    """
    The backward functions
    """

    # The loss function 
    def __loss_function(self, batch, batch_labels, weights=None, with_prediction = True) : 
        """
        Calculating the categorical cross entropy
        """
        logits = self.forward(batch, weights) # The weights should update 
        max_logits = jnp.max(logits, axis= 1, keepdims=True)
        corrected_logits = logits - max_logits
        # Here we do not go to exp logits to keep stability  
        log_probs = corrected_logits - jnp.log(jnp.sum(jnp.exp(corrected_logits), axis=-1, keepdims=True))
        correct_log_probs = log_probs[jnp.arange(logits.shape[0]), batch_labels]
        cce = -correct_log_probs
        mean_cce = jnp.mean(cce)
        if with_prediction :  # Just to make computation faster (we will make sure this is a lot better later on,  separating logit computation )
            exp_logits = jnp.exp(corrected_logits)
            sum_exp_logits = jnp.sum(exp_logits, axis=1, keepdims=True)
            probabilities = exp_logits/sum_exp_logits
            return mean_cce, jnp.argmax(probabilities, axis=1)
        return mean_cce, None 
    
    # The optimizer (SGD with momentum as used in the paper)
    # Let's manage the velocities for each of the weights: 
    def __init_momentum_buffers(self): 
        for layer in self.__weights : 
            self.__velocities[layer] = {}
            for param in self.__weights[layer]: 
                self.__velocities[layer][param] = jnp.zeros_like(self.__weights[layer][param])

    
    def __optimizer_step(self, batch, batch_labels, learning_rate = 1e-3, momentum = 0.09 ):
        """
        Simple SGD
        """ 
        grads = jax.grad(lambda weights: self.__loss_function(batch, batch_labels, weights, False)[0])(self.__weights)        
        for layer in self.__weights: 
            for param in self.__weights[layer]:
                self.__velocities[layer][param] = momentum * self.__velocities[layer][param] + learning_rate * grads[layer][param] # Going in the same direction as previous one
                self.__weights[layer][param] -= self.__velocities[layer][param]

    def __train_data(self, train_generator, learning_rate = 1e-3, momentum=0.09, curr_epoch=1, with_metrics= True): # For one epoch
        train_loss = 0.0
        cpt = 0
        metrics = None 
        with_prediction = False
        if with_metrics : 
            metrics = {
                "train_accuracy" : 0, 
            }
            with_prediction = True
        pbar = tqdm(train_generator, total=self.__total_train_batches, desc=f"Epoch {curr_epoch}") 
        for batch in pbar : 
            images = batch[0] 
            labels = batch[1]
            loss, pred = self.__loss_function(images, labels, with_prediction=with_prediction)
            train_loss += loss
            if with_metrics : 
                metrics["train_accuracy"] += self.__accuracy_unn(pred, labels)
            self.__optimizer_step(images, labels, learning_rate, momentum)
            cpt += batch[0].shape[0]
            if with_metrics :
                pbar.set_postfix(loss = f"{loss/cpt}", accuracy= f"{metrics["train_accuracy"]/cpt:.4f}", refresh= False)
            else : 
                pbar.set_postfix(loss = f"{loss/cpt}", refresh= False)
        if with_metrics : 
            metrics["train_accuracy"] /= cpt
            metrics["train_accuracy"] = metrics["train_accuracy"].item()
            return train_loss/cpt, metrics
        return train_loss/cpt , None # Some normalization 
    
    def __validate_data(self, val_generator, with_metrics = True): 
        val_loss = 0.0 
        cpt = 0 
        metrics = None 
        with_prediction = False 
        if with_metrics : 
            metrics = {
                "val_accuracy" : 0
            }
            with_prediction = True
        for batch in val_generator : 
            images = batch[0]
            labels = batch[1]
            loss, pred = self.__loss_function(images, labels, with_prediction = with_prediction) 
            val_loss += loss
            if with_metrics :
                metrics["val_accuracy"] += self.__accuracy_unn(pred, labels)
            cpt += batch[0].shape[0]
        if with_metrics : 
            metrics["val_accuracy"] /= cpt
            metrics["val_accuracy"] = metrics["val_accuracy"].item()
            return val_loss/cpt, metrics
        return val_loss/cpt, None
    
    # Having these two functions mean the model is already precompiled if constructed
    def train(self, dataloader, epochs = 10, batch_size = 32 , learning_rate= 1e-3, momentum = 0.09 ) : 
        """
        The big training function 
        """
        tgf = next(dataloader.generate_data_as_batch(batch_size)) 
        # First we try a run : 
        self.forward(tgf[0])
        # If you retry to train, all momentum gets done
        self.__init_momentum_buffers()
        history = {
            "train_loss" : [],
            "val_loss" : [],
            "metrics" : {
                "train_accuracy" : [],
                "val_accuracy" : []
            }
        }
        # Now let's loop on each epoch : 
        for epoch in range(epochs) : 
            # Data preparing
            train_generator, val_generator, self.__total_train_batches, self.__total_test_batches = dataloader.generate_data_as_train_test_split(batch_size, shuffle=True, keep_previous=True)
            # 1) Let's train 
            train_loss, tmetrics = self.__train_data(train_generator, learning_rate, momentum, epoch, with_metrics = True)
            history["train_loss"].append(train_loss)
            history["metrics"]["train_accuracy"].append(tmetrics["train_accuracy"])
            # 2) Let's val 
            val_loss, vmetrics = self.__validate_data(val_generator, with_metrics=True)
            history["val_loss"].append(val_loss)
            history["metrics"]["val_accuracy"].append(vmetrics["val_accuracy"])
            print("Epoch :",epoch,"| Training Loss :",train_loss ,"| Validation Loss :",val_loss)  
            print("Current Metrics: | Training Accuracy:", tmetrics["train_accuracy"], "| Validation Accuracy:", vmetrics["val_accuracy"])
        return history     
    
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
    
    """
    The Metrics (Better to construct a class outside but for now let's do it like this)
    """

    def __accuracy_unn(self, prediction, labels ) : # Computed within the batch 
        return jnp.sum(prediction == labels)

    """
    Model Saving
    """

    def save(self, filename) :
        """
        Saving the weights: 
        """
        # Bundling necessary variables
        model_state = {
            "weights": self.__weights,
            "shapes": self.__shapes,
            "config": self.__config,
        }
        try : 
            with open(filename, 'wb') as f:
                pickle.dump(model_state, f)
            print(f"Model weights and configuration successfully saved to: **{os.path.abspath(filename)}**")
        except Exception as e : 
            print(f"Error saving state: {e}")

    def load(self, filename) : 
        if not os.path.exists(filename):
            print(f"Error: State file not found at {os.path.abspath(filename)}")
            return False
        try:
            with open(filename, 'rb') as f:
                loaded_state = pickle.load(f)
                if "weights" in loaded_state:
                    self.__weights = loaded_state["weights"]
                if "shapes" in loaded_state:
                    self.__shapes = loaded_state["shapes"]
                if "config" in loaded_state:
                    self.__config = loaded_state["config"]
                self.__velocities = {} 
                print(f"Full model state (weights, shapes, and config) successfully loaded from: **{os.path.abspath(filename)}**")
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False