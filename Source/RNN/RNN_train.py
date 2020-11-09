
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def simple_rnn(t, t_train, s_train, vocab_size):
    tf.keras.backend.clear_session()
    # hyper parameters
    epochs = 5
    batch_size = 32
    embed_dim = 16
    units = 128

    model = tf.keras.Sequential([
        L.Embedding(vocab_size, embed_dim, input_length=t.shape[1]),
        L.SimpleRNN(units, return_sequences=True),
        L.GlobalMaxPool1D(),
        L.Dropout(0.4),
        L.Dense(64, activation="relu"),
        L.Dropout(0.4),
        L.Dense(5)])

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    model.summary()

    trained = model.fit(t_train, s_train, epochs=epochs, validation_split=0.12, batch_size=batch_size)
    model.save('./Resources/Models/RNN/simpleRNN_bcrossentropy')

    return model


def bidirect_rnn(t, t_train, s_train, vocab_size):
    tf.keras.backend.clear_session()
    # hyper parameters
    epochs = 5
    batch_size = 32
    embed_dim = 16
    units = 128

    model = tf.keras.Sequential([
        L.Embedding(vocab_size, embed_dim, input_length=t.shape[1]),
        L.Bidirectional(L.SimpleRNN(units, return_sequences=True)),
        L.GlobalMaxPool1D(),
        L.Dropout(0.4),
        L.Dense(64, activation="relu"),
        L.Dropout(0.4),
        L.Dense(5)])

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    model.summary()

    trained = model.fit(t_train, s_train, epochs=epochs, validation_split=0.12, batch_size=batch_size)
    model.save('./Resources/Models/RNN/biRNN_bcrossentropy')

    return model



