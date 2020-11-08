import tensorflow as tf
from tensorflow import keras
from transformers import TFBertForSequenceClassification

from Source.Utility.constants import MAXIMUM_SENTENCE_LENGTH, NUMBER_OF_WORDS


class Generator:
    """
    No state class that contains functions which generate models
    """

    @staticmethod
    def generate_simple_gru_model(hyperparameters):
        """
        Function that generates a simple GRU model from hyperparameters
        :return: The GRU model
        """
        model = keras.Sequential()

        hp_output_dim = hyperparameters.Int('embedding_output_dim', min_value=8, max_value=32, step=8)
        model.add(keras.layers.Embedding(
            input_dim=NUMBER_OF_WORDS,
            output_dim=hp_output_dim,
            input_length=MAXIMUM_SENTENCE_LENGTH
        ))

        hp_hidden_units = hyperparameters.Int('gru_hidden_units', min_value=8, max_value=128, step=8)
        hp_dropout_rate = hyperparameters.Float('gru_dropout', min_value=0, max_value=0.5, step=0.1)
        model.add(keras.layers.GRU(
            units=hp_hidden_units,
            dropout=hp_dropout_rate
        ))

        model.add(keras.layers.Dense(5))

        hp_learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def generate_bidirectional_gru_model(hyperparameters):
        """
        Function that generates a bidirectional GRU model from hyperparameters
        :return: The GRU model
        """
        model = keras.Sequential()

        hp_output_dim = hyperparameters.Int('embedding_output_dim', min_value=8, max_value=32, step=8)
        model.add(keras.layers.Embedding(
            input_dim=NUMBER_OF_WORDS,
            output_dim=hp_output_dim,
            input_length=MAXIMUM_SENTENCE_LENGTH
        ))

        hp_hidden_units = hyperparameters.Int('gru_hidden_units', min_value=8, max_value=128, step=8)
        hp_dropout_rate = hyperparameters.Float('gru_dropout', min_value=0, max_value=0.5, step=0.1)
        model.add(keras.layers.Bidirectional(keras.layers.GRU(
            units=hp_hidden_units,
            dropout=hp_dropout_rate
        )))

        model.add(keras.layers.Dense(5))

        hp_learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model
