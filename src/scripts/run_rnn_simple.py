#######################################################################################################################

# This script performs the training for a simple RNN architecture and cross validates it using kerastunner.

# The data is read in the form of test train sets and the hyper tuner tunes the model training for different parameters
# defined in 'generation.py' script. The best hyper parameters are saved and used to train the model.

# The results are extracted and saved for the tuned model of simple RNN architecture.

#######################################################################################################################
from src.util.data import get_train_test_sets
from src.util.extraction import ResultsExtractor

from src.util.generation import Generator
import kerastuner as kt
import tensorflow as tf
from src.util.constants import *

# enable gpu processing for the tuning and training
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# set default paths using 'constants.py' definitions for the script
model_path = RNN_MODEL_PATH + '/Simple'
results_path = RNN_RESULTS_PATH + '/Simple'


#  get train and test sets for training
x_train, y_train, x_test, y_test = get_train_test_sets()


###########################################Hyperparameter Tuning
tuner = kt.Hyperband(Generator.generate_rnn_simple,
                     objective='val_accuracy', max_epochs=5, factor=2, directory=model_path,
                     project_name='rnn_simple_hp')

tuner.search(x_train, y_train, epochs=5, validation_split=0.10)

tuned_hp = tuner.get_best_hyperparameters()[0]

rnn_s = Generator.generate_rnn_simple(tuned_hp)

checkpoint_path = model_path + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# ########################## simple RNN model training#############################
rnn_s.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.10)
rnn_s.save(model_path + '/mysimpleRNN_bcrossentropy')

# rnn_s = kr.load_model(model_path + '/mysimpleRNN_bcrossentropy');


##########################################Evaluation
predictions = rnn_s.predict(x=x_test)
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(results_path)
results.save_confusion_matrix(results_path)
results.save_per_class_metrics(results_path)
results.save_macro_averaged_metrics(results_path)

# Save the best hyperparameters
with open(results_path + '/' + 'simple_rnn_tuned_hps.txt', 'w') as file:
    file.write('embedding_dimension: {:.5f}\n'.format(tuned_hp.get('embedding_dimension')))
    file.write('rnn_units: {:.5f}\n'.format(tuned_hp.get('rnn_units')))
    file.write('dropout_rate1: {:.5f}\n'.format(tuned_hp.get('dropout_rate1')))
    file.write('dropout_rate2: {:.5f}\n'.format(tuned_hp.get('dropout_rate2')))
    file.write('learning_rate: {:.5f}\n'.format(tuned_hp.get('learning_rate')))
    file.close()
