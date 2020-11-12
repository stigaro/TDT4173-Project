import os
import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.util.constants import *
from src.util.extraction import ResultsExtractor
from src.util.generation import Generator
from src.util.loading import load_simple_sentence_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Load the dataset
train_x, train_y, test_x, test_y = load_simple_sentence_dataset()

# Tokenize the sentences
tokenizer = Tokenizer(num_words=NUMBER_OF_WORDS, oov_token=True)
tokenizer.fit_on_texts(train_x.tolist())
train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

# Adds padding to tokenized sentences not at max length
# and reshape labels to work with tensorflow
train_x = pad_sequences(train_x, maxlen=MAXIMUM_SENTENCE_LENGTH)
train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
test_x = pad_sequences(test_x, maxlen=MAXIMUM_SENTENCE_LENGTH)
test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

# Define a hyperband for searching hyperparameters
# (If search has already been done it will simply load from the directory)
tuner = kt.Hyperband(
    Generator.generate_bidirectional_gru_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=2,
    directory=PATH_TO_MODEL_GRU_BIDIRECTIONAL_HYPERPARAMETER,
    project_name=HYPER_PARAMETER_PROJECT_NAME
)

# Perform search for best hyperparameters (If not already done)
tuner.search(train_x, train_y, epochs=5, validation_split=0.10)

# Get the optimal hyperparameters, and use those to create a model for training
best_hyperparameters = tuner.get_best_hyperparameters()[0]
model = Generator.generate_bidirectional_gru_model(best_hyperparameters)

# Create a callback that saves the model's weights every epoch
checkpoint_path = PATH_TO_MODEL_GRU_BIDIRECTIONAL_CHECKPOINTS + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Fit the model and save it
model.fit(train_x, train_y, epochs=5, batch_size=64, callbacks=[cp_callback], validation_split=0.10)
model.save(PATH_TO_MODEL_GRU_BIDIRECTIONAL_SAVE)

# Save the results
predictions = model.predict(x=test_x)
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(PATH_TO_RESULT_GRU_BIDIRECTIONAL)
results.save_confusion_matrix(PATH_TO_RESULT_GRU_BIDIRECTIONAL)
results.save_per_class_metrics(PATH_TO_RESULT_GRU_BIDIRECTIONAL)
results.save_macro_averaged_metrics(PATH_TO_RESULT_GRU_BIDIRECTIONAL)
results.save_best_hyperparameters(best_hyperparameters, PATH_TO_RESULT_GRU_BIDIRECTIONAL)
