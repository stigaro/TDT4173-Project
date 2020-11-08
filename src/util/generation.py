import tensorflow as tf
from tensorflow import keras
from transformers import TFBertForSequenceClassification

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
        model.add(keras.layers.Dense(5))
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
        model.add(keras.layers.Dense(5))
        return model

    @staticmethod
    def generate_transformer_bert_model():
        transformer = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)
        input_ids = tf.keras.layers.Input(shape=(MAXIMUM_SENTENCE_LENGTH,), name='input_token', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(MAXIMUM_SENTENCE_LENGTH,), name='masked_token', dtype='int32')
        transformer_output = transformer(input_ids, input_masks_ids)[0]
        transformer_output = tf.keras.layers.Dense(5, activation='sigmoid')(transformer_output)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=transformer_output)
        return model
