import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy
import pandas
from keras import layers, regularizers


class AutoEncoder:
    """
    Class that contains the abstract methods of an AutoEncoder (Keras)
    """

    def __init__(self, input_size, min_size):
        """
        Constructor of a generic Classifier
        :param input_size: size of the input
        """
        self.trained = False
        self.autoencoder = self.init_autoencoder(input_size, min_size)

    def init_autoencoder(self, input_size, min_size):

        keras_input = keras.Input(shape=(input_size,))

        encoded, decoded = self.build_autoencoder(input_size, min_size, keras_input)

        # This model maps an input to its reconstruction
        autoencoder = keras.Model(keras_input, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        return autoencoder

    def build_autoencoder(self, input_size, min_size):
        print("Warning: Method not overridden")
        pass

    def fit(self, x_train, epochs=50, batch_size=256, train_val_split=0.8, verbose=0):
        """
        Fits an AutoEncoder
        :param train_val_split: split between train and validation set
        :param batch_size: batch size
        :param epochs: number of epochs
        :param x_train: feature set
        """
        if isinstance(x_train, pandas.DataFrame):
            data_set = copy.deepcopy(x_train.values)
        else:
            data_set = copy.deepcopy(x_train)
        numpy.random.shuffle(data_set)
        split_index = int(train_val_split*data_set.shape[0])
        train_set, val_set = data_set[:split_index, :], data_set[split_index:, :]
        self.autoencoder.fit(train_set, train_set, epochs=epochs, batch_size=batch_size,
                             shuffle=True, validation_data=(val_set, val_set), verbose=verbose)
        self.trained = True

    def is_trained(self):
        """
        Flags if train was executed
        :return: True if trained, False otherwise
        """
        return self.trained

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :return: autoencoded input
        """
        decoded_test = self.autoencoder.predict(x_test)
        mae_loss = keras.losses.mae(decoded_test, x_test).numpy()
        return decoded_test, mae_loss


class SingleAutoEncoder(AutoEncoder):

    def build_autoencoder(self, input_size, min_size, keras_input):

        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(min_size, activation='relu')(keras_input)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(input_size, activation='sigmoid')(encoded)

        return encoded, decoded


class SingleSparseAutoEncoder(AutoEncoder):

    def build_autoencoder(self, input_size, min_size, keras_input):
        # Add a Dense layer with a L1 activity regularizer
        encoded = layers.Dense(min_size, activation='relu',
                               activity_regularizer=regularizers.l1(10e-5))(keras_input)
        decoded = layers.Dense(input_size, activation='sigmoid')(encoded)

        return encoded, decoded


class DeepAutoEncoder(AutoEncoder):

    def build_autoencoder(self, input_size, min_size, keras_input):
        # Add a Dense layer with a L1 activity regularizer
        step = input_size - min_size/3
        encoded = layers.Dense(min_size + 2*step, activation='relu')(keras_input)
        encoded = layers.Dense(min_size + step, activation='relu')(encoded)
        encoded = layers.Dense(min_size, activation='relu')(encoded)

        decoded = layers.Dense(min_size + step, activation='relu')(encoded)
        decoded = layers.Dense(min_size + 2*step, activation='relu')(decoded)
        decoded = layers.Dense(input_size, activation='sigmoid')(decoded)

        return encoded, decoded
