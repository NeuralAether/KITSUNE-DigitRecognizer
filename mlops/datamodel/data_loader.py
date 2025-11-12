"""
The data loader class 
"""

from config import * 

class DM_Loader : 
    """
    This class loads the csv file of images and can manage the getters for the need of DL 
    """
    def __init__(self):
        # Some private vals: 
        self.__image_data = None 
        self.__type = None
        self.__shuffled_indices = None

    """
    Setters 
    """

    def set_new_data(self, data : pd.DataFrame) -> None: 
        # Data Check to be done later (has to resemble the example csv)
        self.__image_data = data 
        if "label" in self.__image_data.columns : 
            self.__type = "train"
        else : 
            self.__type = "test"

    def set_new_data_from_config(self, type= "train") -> None: 
        # Data build:
        assert type in ["train","test"], "Should either be train or test"
        self.__type = type
        self.__image_data = pd.read_csv(PROJECT_HEADERS["cwd_raw"] + f"{type}.csv")            

    """
    Getters
    """

    def get_full_data(self) -> pd.DataFrame : 
        return self.__image_data 
    
    def get_data_at_id(self, id: int) -> Tuple[jnp.ndarray, jnp.ndarray|None] :
        __l = self.__image_data.iloc[id] 
        if self.__type == "train" : 
            y = __l["label"]
            X = __l.loc["pixel0":].values.reshape(28, 28)/255
            return jnp.expand_dims(X, axis=-1), jnp.array(y)
        else:
            X = __l.values.reshape(28, 28)/255
            return jnp.expand_dims(X, axis=-1), None
    
    """
    Generators
    """

    def generate_data_as_element(self) : 
        for i in range(self.__image_data.shape[0]) : 
            yield self.get_data_at_id(i)

    def generate_data_as_batch(self, batch_size : int) : 
        n = self.__image_data.shape[0]
        for start in range(0, n, batch_size) : 
            end = min(start + batch_size, n)
            batch = self.__image_data.iloc[start:end]
            if self.__type == "train" : 
                y_batch = batch["label"].values
                X_batch = batch.loc[:, "pixel0":].values.reshape(-1,28,28)/255.0
                yield jnp.expand_dims(X_batch, axis=-1), jnp.array(y_batch)
            else : 
                X_batch = batch.values.reshape(-1, 28,28)/255.0
                yield jnp.expand_dims(X_batch, axis=-1), None
        
    def generate_data_as_train_test_split(self, batch_size: int, val_split= 0.2, shuffle = True, keep_previous=True): 
        assert val_split>0 and val_split<1, "Invalid train_test_split value, should be between 0 and 1, strictly"
        n = self.__image_data.shape[0]
        val_size = int(n * val_split) 
        train_size = n - val_size
        
        if shuffle and (not keep_previous or self.__shuffled_indices is None): 
            self.__shuffled_indices = np.random.permutation(n)
        else : 
            self.__shuffled_indices = np.arange(n)
        train_data = self.__image_data.iloc[self.__shuffled_indices[:train_size]].copy()
        val_data = self.__image_data.iloc[self.__shuffled_indices[train_size:]].copy()
        # The generators :
         # Generator for Training Data Batches
        def train_generator():
            for start in range(0, train_size, batch_size):
                end = min(start + batch_size, train_size)
                batch = train_data.iloc[start:end]
                
                # Assuming 'label' exists for training/validation data
                y_batch = batch["label"].values
                X_batch = batch.loc[:, "pixel0":].values.reshape(-1, 28, 28) / 255.0
                
                # Yield data formatted for JAX
                yield jnp.expand_dims(X_batch, axis=-1), jnp.array(y_batch)
        # Generator for Validation Data Batches
        def val_generator():
            for start in range(0, val_size, batch_size):
                end = min(start + batch_size, val_size)
                batch = val_data.iloc[start:end]
                
                # Assuming 'label' exists for training/validation data
                y_batch = batch["label"].values
                X_batch = batch.loc[:, "pixel0":].values.reshape(-1, 28, 28) / 255.0
                
                # Yield data formatted for JAX
                yield jnp.expand_dims(X_batch, axis=-1), jnp.array(y_batch)
        return train_generator() , val_generator() , int(train_size/batch_size) + int(train_size%batch_size >0) , int(val_size/batch_size)+int(val_size%batch_size >0) 