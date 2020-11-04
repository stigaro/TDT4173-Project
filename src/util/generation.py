from tensorflow import keras

from src.util.constants import MAXIMUM_SENTENCE_LENGTH


class Generator:
    """
    No state class that contains functions which generate models
    """

    @staticmethod
    def generate_simple_gru_model():
        """
        Function that generates a simple GRU model
        :param maximum_sentence_length: The maximum possible length of a sentence
        :return: The GRU model
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(10000, 16, input_length=MAXIMUM_SENTENCE_LENGTH))
        model.add(keras.layers.GRU(64, dropout=0.2))
        model.add(keras.layers.Dense(5, activation='softmax'))
        return model

    @staticmethod
    def generate_bidirectional_gru_model():
        """
        Function that generates a bidirectional GRU model
        :param maximum_sentence_length: The maximum possible length of a sentence
        :return: The GRU model
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(10000, 16, input_length=MAXIMUM_SENTENCE_LENGTH))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(64, dropout=0.2)))
        model.add(keras.layers.Dense(5, activation='sigmoid'))
        return model
