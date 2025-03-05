from typing import Dict
from keras.layers import Input, Dense
from keras.models import Model
from mnist_dataset_class import Dataset

class Autoencoder:
    def __init__(self, num_hidden_units: int) -> None:
        self.input_shape: int = 784
        self.encoding_activation_function: str = 'relu'
        self.decoding_activation_function: str = 'sigmoid'
        self.optimizer: str = 'adadelta'
        self.loss_function: str = 'mean_squared_error'
        self.num_hidden_units: int = num_hidden_units
        self.num_epochs: int = 50
        self.batch_size: int = 256
        self.shuffle_training_data: bool = True
        self.dataset: Dataset = Dataset(self.input_shape)

        self.train_status: bool = False
        self.model: Model = self.create_autoencoder()
        self.training_history: Dict = None
    
    def create_autoencoder(self) -> Model:
        """
        Initializes a 3-layer Autoencoder model.
        """
        # Create 3-layer autoencoder
        input_img = Input(shape=(self.input_shape,))
        encoded_input = Dense(self.num_hidden_units, activation = self.encoding_activation_function)(input_img)
        decoded_input = Dense(self.input_shape, activation = self.decoding_activation_function)(encoded_input)
        autoencoder_model = Model(input_img, decoded_input)

        # Configure model with chosen loss function and optimizer
        autoencoder_model.compile(optimizer = self.optimizer, loss = self.loss_function)

        return autoencoder_model
    
    def train_model(self) -> None:
        """
        Train model using dataset, and caches training history.
        """
        training_data = self.dataset.training_dataset
        validation_data = self.dataset.validation_dataset
        training_history = self.model.fit(training_data, training_data, epochs=self.num_epochs, batch_size=self.batch_size, shuffle=self.shuffle_training_data, validation_data=(validation_data, validation_data))

        self.train_status = True
        self.training_history = training_history

    def eval_model(self, test_digit: int) -> float:
        """
        Returns MSE for the prediction of specified digit using the testing dataset.
        """
        if not self.train_status:
            raise ValueError("Model has not been trained.")

        # Filter testing dataset to test digit
        test_dataset = self.dataset.testing_dataset
        test_digit_dataset = test_dataset[(self.dataset.testing_labels == test_digit)]

        mse = self.model.evaluate(test_digit_dataset, test_digit_dataset)
        
        return mse