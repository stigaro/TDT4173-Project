import os
import pandas as pd
import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.util.constants import *
from src.util.data import get_train_test_sets
from src.util.extraction import ResultsExtractor
from src.util.generation import Generator
from src.util.loading import load_simple_sentence_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Load the dataset
train_x, train_y, test_x, test_y = get_train_test_sets()

# Define a hyperband for searching hyperparameters
# (If search has already been done it will simply load from the directory)
tuner = kt.Hyperband(
    Generator.generate_mlp_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=2,
    directory=PATH_TO_MODEL_MLP_HYPERPARAMETER,
    project_name=HYPER_PARAMETER_PROJECT_NAME
)

# Perform search for best hyperparameters (If not already done)
tuner.search(train_x, train_y, epochs=5, validation_split=0.10)

# Get the optimal hyperparameters, and use those to create a model for training
best_hyperparameters = tuner.get_best_hyperparameters()[0]
model = Generator.generate_simple_gru_model(best_hyperparameters)

# Create a callback that saves the model's weights every epoch
checkpoint_path = PATH_TO_MODEL_MLP_CHECKPOINTS + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Fit the model and save it
model.fit(train_x, train_y, epochs=5, batch_size=64, callbacks=[cp_callback], validation_split=0.10)
model.save(PATH_TO_MODEL_MLP_SAVE)

# Save the results
predictions = model.predict(x=test_x)
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(PATH_TO_RESULT_MLP)
results.save_confusion_matrix(PATH_TO_RESULT_MLP)
results.save_per_class_metrics(PATH_TO_RESULT_MLP)
results.save_macro_averaged_metrics(PATH_TO_RESULT_MLP)

# Save the best hyperparameters
with open(PATH_TO_RESULT_MLP + '/' + 'best_hyperparameters.txt', 'w') as file:
    file.write('hidden_layer_1_units: {:.5f}\n'.format(best_hyperparameters.get('hidden_layer_1_units')))
    file.write('hidden_layer_2_units: {:.5f}\n'.format(best_hyperparameters.get('hidden_layer_2_units')))
    file.write('hidden_layer_3_units: {:.5f}\n'.format(best_hyperparameters.get('hidden_layer_3_units')))
    file.write('hidden_layer_4_units: {:.5f}\n'.format(best_hyperparameters.get('hidden_layer_4_units')))
    file.write('dropout_Rate: {:.5f}\n'.format(best_hyperparameters.get('dropout_Rate')))
    file.write('learning_rate: {:.5f}\n'.format(best_hyperparameters.get('learning_rate')))
    file.close()
