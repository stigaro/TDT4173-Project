import os
import numpy as np
import tensorflow as tf
import kerastuner as kt
from sklearn.pipeline import Pipeline

import src.util.constants as CONST
from src.util.extraction import ResultsExtractor
from src.util.generation import Generator
from src.util.data import get_train_test_sets

# GPU specific configs
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

x_train, y_train, x_test, y_test = get_train_test_sets()

# Define a hyperband for searching hyperparameters
# (If search has already been done it will simply load from the directory)
tuner = kt.Hyperband(
    Generator.generate_lstm_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=2,
    directory=CONST.LSTM_MODEL_PATH,
    project_name=CONST.HYPER_PARAMETER_PROJECT_NAME
)

# Perform search for best hyperparameters (If not already done)
tuner.search(x_train, y_train, epochs=5, validation_split=0.10)

# Get the optimal hyperparameters, and use those to create a model for training
best_params = tuner.get_best_hyperparameters()[0]
model = Generator.generate_lstm_model(best_params)

# Create a callback that saves the model's weights every epoch
os.makedirs(CONST.LSTM_CHECKPOINT_PATH, exist_ok=True)
checkpoint_path = CONST.LSTM_CHECKPOINT_PATH + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Fit the model and save it
model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[cp_callback], validation_split=0.10)
model.save(CONST.LSTM_FINAL_PATH)

# Save the results
predictions = model.predict(x=x_test)
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(CONST.LSTM_RESULTS_PATH)
results.save_confusion_matrix(CONST.LSTM_RESULTS_PATH)
results.save_per_class_metrics(CONST.LSTM_RESULTS_PATH)
results.save_macro_averaged_metrics(CONST.LSTM_RESULTS_PATH)

# Save the best hyperparameterss
best_param_path = os.path.join(CONST.LSTM_RESULTS_PATH, 'best_hyperparameters.txt')
with open(best_param_path, 'w') as f:
    for param in ['embedding_output_dim', 'lstm_hidden_units',
                  'first_dropout_rate', 'second_dropout_rate',
                  'learning_rate']:
        f.write(param + ': {:.5f}\n'.format(best_params.get(param)))
