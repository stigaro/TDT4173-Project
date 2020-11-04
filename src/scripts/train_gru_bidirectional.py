import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.util.constants import *
from src.util.generation import Generator
from src.util.loading import load_simple_sentence_dataset

# Load the dataset
train_x, train_y, test_x, test_y = load_simple_sentence_dataset()

# Create the model
model = Generator.generate_bidirectional_gru_model()

# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = PATH_TO_MODEL_GRU_BIDIRECTIONAL_CHECKPOINTS + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model and save it
model.fit(train_x, train_y, epochs=10, batch_size=10, callbacks=[cp_callback], validation_split=0.2)
model.save(PATH_TO_MODEL_GRU_BIDIRECTIONAL_SAVE)
