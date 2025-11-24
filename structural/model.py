"""
The structure for a model in jax: To make it easier to code
"""

from config import * 
class JaxModel : 
    """
    Implementation of a model 
    """
    def __init__(self, **kwargs): 
        # -------------- For all --------------------------
        # Layer names : 
        self.layers = [] # Layers are coded in a custom way 
        # FOR NOW : layers are defined directly inside the class.

        # -------------- Compilation Details ---------------
        self.loss_function = None 
        self.optimizer = None
        self.is_compiled = False
        # -------------- Training Details ------------------
        self.callables = kwargs.get("callables",{})  # Functions to call (name and function)
        # -------------- Some kwargs -----------------------
        self.__kwargs = kwargs

    """
    The forward functions 
    """

    def construct(self, batch: jnp.ndarray, **kwargs): 
        """
        To run specifically to data
        """
        __aux = batch
        for layer in self.layers: 
            layer.construct(__aux) 
            __aux = layer(__aux)
        pass

    def forward(self, batch: jnp.ndarray, **kwargs): 
        """
        The forward pass: (To code here)
        """
        __res = batch
        for layer in self.layers : 
            __res = layer(__res)
        return __res
     
    def predict(self, batch: jnp.ndarray, **kwargs) :
        """
        A special form of forward for inference
        """
        __response = self.forward(batch)
        # Do something here
        return __response

    """
    The compiler functions 
    """

    def compile(self, loss_function, optimizer) : 
        self.loss_function = loss_function 
        self.optimizer = optimizer 
        self.is_compiled = True

    """
    Weight setting and getting
    """

    def copy(self) : 
        """
        Copies the model 
        """
        new_model = JaxModel(**self.__kwargs)
        for layer in self.layers : 
            new_model.layers.append(layer.copy())
        return new_model

    def get_weights(self) : 
        weights = {}
        for layer in self.layers : 
            weights[layer.name] = layer.get_weights()
        return weights
    
    def set_weights(self, weights : dict) :
        set_weights = 0
        for layer in self.layers : 
            if layer.name in weights : 
                layer.set_weights(weights[layer.name]["weights"])
                set_weights += 1
        if set_weights != len(weights) : 
            print(f"Warning : Only {set_weights} out of {len(weights)} layers were set. Check layer names. ")
        return set_weights

    """
    The fitting 
    """

    def fit(self, train_generator, train_num_batches=None, val_generator = None , val_num_batches = None , **kwargs ): 
        """
        Training 
        """
        # Check compilation
        if not self.is_compiled :
            raise Exception("Model has not been compiled yet. Please compile before fitting. ")
        # Fitting 
        train_loss_history = []
        val_loss_history = []
        epochs = kwargs.get("num_epochs", 1) # Has to appear in kwargs
        training_num_batches = train_num_batches
        validation_num_batches = val_num_batches
        for batch in train_generator : 
            images_batch, labels_batch = batch
            # To be coded : The training step
            loss = self.loss_function(self, images_batch, labels_batch)
            train_loss_history.append(loss)
            # Optimizer step
            new_weights = self.optimizer.step(self, images_batch, labels_batch)  # To finish later      


    """
    The save and load to be done later
    """

    """
    Summary 
    """
    def get_summary(self) : 
        # We get all information from summaries 
        rows = []
        for i, layer in enumerate(self.layers) : 
            item = layer.get_summary()
            details = item["weights"].get("trainable_details")
            if details:
                for k, v in details.items():
                    rows.append({
                        "id" : i, 
                        "layer_id": item["id"],
                        "layer_name": item["name"],
                        "shape": item["weights"].get("shape"),
                        "trainable": item["weights"].get("trainable"),
                        "detail_key": k,
                        "detail_value": v
                    })
            else:
                # add empty record for trainable 0 case
                rows.append({
                    "id" : i, 
                    "layer_id": item["id"],
                    "layer_name": item["name"],
                    "shape": item["weights"].get("shape"),
                    "trainable": item["weights"].get("trainable"),
                    "detail_key": "",
                    "detail_value": 0
                })
        df = pd.DataFrame(rows).set_index(["id","layer_id", "layer_name", "shape", "trainable", "detail_key"])
        print("The number of trainable parameters is :", int(df["detail_value"].sum()) )
        return df