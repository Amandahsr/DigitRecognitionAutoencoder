import numpy as np
from keras.datasets import mnist
import ssl

class Dataset:
    def __init__(self, model_input_shape: int) -> None:
        self.dataset_dtype: str = 'float32'
        self.image_shape: int = model_input_shape

        self.training_dataset: np.ndarray = None
        self.training_labels: np.ndarray = None
        self.validation_dataset: np.ndarray = None
        self.testing_dataset: np.ndarray = None
        self.testing_labels: np.ndarray = None
        self.load_datasets()
    
    def load_datasets(self) -> None:
        """
        Loads training, validation and testing dataset from MNIST.
        """
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load MNIST dataset
        (training_dataset, training_labels), (testing_dataset, testing_labels) = mnist.load_data()

        # Normalize MNIST dataset and flatten them to valid model input
        training_dataset = training_dataset.astype(self.dataset_dtype) / 255
        training_dataset = training_dataset.reshape((len(training_dataset), self.image_shape))
        testing_dataset = testing_dataset.astype(self.dataset_dtype) / 255
        testing_dataset = testing_dataset.reshape((len(testing_dataset), self.image_shape))

        # Use testing dataset as validation dataset
        self.training_dataset = training_dataset
        self.training_labels = training_labels
        self.validation_dataset = testing_dataset
        self.testing_dataset = testing_dataset
        self.testing_labels = testing_labels
