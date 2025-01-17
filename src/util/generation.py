from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.python.keras.layers import SimpleRNN
from keras.optimizers import Adam

from src.util.constants import MAXIMUM_SENTENCE_LENGTH, NUMBER_OF_WORDS, NUM_CLASSES


class Generator:
    """
    No state class that contains functions which generate models
    """

    @staticmethod
    def generate_rnn_simple(hp):

        # parameters to be tuned
        hp_embed_dim = hp.Int('embedding_dimension', min_value=8, max_value=64, step=8) # embed_dim = 16
        hp_dropout1 = hp.Float('dropout_rate1', min_value=0, max_value=0.7, step=0.1)
        hp_dropout2 = hp.Float('dropout_rate2', min_value=0, max_value=0.7, step=0.1)
        hp_units = hp.Int('rnn_units', min_value=8, max_value=256, step=8) # units = 128
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

        #model build
        rnn_s = keras.Sequential();
        rnn_s.add(Embedding(NUMBER_OF_WORDS+1, hp_embed_dim, input_length= MAXIMUM_SENTENCE_LENGTH))
        rnn_s.add(Dropout(hp_dropout1))
        rnn_s.add(SimpleRNN(hp_units, activation= 'softmax'))
        rnn_s.add(Dropout(hp_dropout2))
        rnn_s.add(Dense(5))

        rnn_s.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer= Adam(hp_learning_rate),
                      metrics=['accuracy'])
        # rnn_s.summary()

        return rnn_s;

    @staticmethod
    def generate_rnn_bidirect(hp):

        # parameters to be tuned
        hp_embed_dim = hp.Int('embedding_dimension', min_value=8, max_value=64, step=8) # embed_dim = 16
        hp_dropout1 = hp.Float('dropout_rate1', min_value=0, max_value=0.7, step=0.1)
        hp_dropout2 = hp.Float('dropout_rate2', min_value=0, max_value=0.7, step=0.1)
        hp_units = hp.Int('rnn_units', min_value=8, max_value=256, step=8) # units = 128
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

        # model build
        rnn_b = keras.Sequential();
        rnn_b.add(Embedding(NUMBER_OF_WORDS + 1, hp_embed_dim, input_length=MAXIMUM_SENTENCE_LENGTH))
        rnn_b.add(Dropout(hp_dropout1))
        rnn_b.add(keras.layers.Bidirectional(SimpleRNN(hp_units, activation='softmax')))
        rnn_b.add(Dropout(hp_dropout2))
        rnn_b.add(Dense(5))

        rnn_b.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer= Adam(hp_learning_rate),
                      metrics=['accuracy'])
        # rnn_b.summary()

        return rnn_b;

    @staticmethod
    def generate_lstm_model(hp):
        """
        Samples a compiled lstm keras model.

        Args:
            hp: hyperparameter object passed in by Keras
            Hyperband api.
        """

        lstm = keras.Sequential()

        lstm.add(Embedding(NUMBER_OF_WORDS + 1,
                           hp.Int('embedding_output_dim', min_value=8, max_value=32, step=8),
                           input_length=MAXIMUM_SENTENCE_LENGTH))

        lstm.add(LSTM(
            units=hp.Int('lstm_hidden_units', min_value=8, max_value=128, step=8),
            dropout=hp.Float('lstm_dropout', min_value=0, max_value=0.5, step=0.1)
        ))

        lstm.add(Dense(NUM_CLASSES))

        lstm.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
            ),
            metrics=['accuracy']
        )

        return lstm

    @staticmethod
    def generate_simple_gru_model(hyperparameters):
        """
        Function that generates a simple GRU model from hyperparameters
        :return: The GRU model
        """
        model = keras.Sequential()

        hp_output_dim = hyperparameters.Int('embedding_output_dim', min_value=8, max_value=32, step=8)
        model.add(keras.layers.Embedding(
            input_dim=NUMBER_OF_WORDS + 1,
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
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
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
            input_dim=NUMBER_OF_WORDS + 1,
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
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def generate_mlp_model(hyperparameters):
        """
        Function that generates a bidirectional GRU model from hyperparameters
        :return: The GRU model
        """
        model = keras.Sequential()

        model.add(keras.Input(shape=MAXIMUM_SENTENCE_LENGTH))

        model.add(keras.layers.Dense(
            hyperparameters.Int('hidden_layer_1_units', min_value=128, max_value=1024, step=128),
            activation=keras.activations.relu
        ))
        model.add(keras.layers.Dropout(hyperparameters.Float('dropout_Rate', min_value=0, max_value=0.4, step=0.1)))

        model.add(keras.layers.Dense(
            hyperparameters.Int('hidden_layer_2_units', min_value=64, max_value=512, step=64),
            activation=keras.activations.relu
        ))
        model.add(keras.layers.Dropout(hyperparameters.Float('dropout_Rate', min_value=0, max_value=0.4, step=0.1)))

        model.add(keras.layers.Dense(
            hyperparameters.Int('hidden_layer_3_units', min_value=32, max_value=256, step=32),
            activation=keras.activations.relu
        ))
        model.add(keras.layers.Dropout(hyperparameters.Float('dropout_Rate', min_value=0, max_value=0.4, step=0.1)))

        model.add(keras.layers.Dense(
            hyperparameters.Int('hidden_layer_4_units', min_value=16, max_value=128, step=16),
            activation=keras.activations.relu
        ))
        model.add(keras.layers.Dropout(hyperparameters.Float('dropout_Rate', min_value=0, max_value=0.4, step=0.1)))

        model.add(keras.layers.Dense(5))

        hp_learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model
