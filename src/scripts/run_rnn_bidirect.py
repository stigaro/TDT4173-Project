from src.util.data import get_train_test_sets
from src.util.extraction import ResultsExtractor
from src.util.constants import *
import kerastuner as kt
import tensorflow as tf
from src.util.generation import Generator

model_path = RNN_MODEL_PATH + '/Bidirect'
results_path = RNN_RESULTS_PATH + '/Bidirect'


#load data
x_train, y_train, x_test, y_test = get_train_test_sets()

###########################################Hyper Parameter tuning
tuner = kt.Hyperband(Generator.generate_rnn_bidirect,
                     objective='val_accuracy', max_epochs=5, factor=2, directory=model_path,
                     project_name='rnn_bidirect_hp')

tuner.search(x_train, y_train, epochs=5, validation_split=0.10)

tuned_hp = tuner.get_best_hyperparameters()[0]

rnn_b = Generator.generate_rnn_bidirect(tuned_hp)

checkpoint_path = model_path + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Bidirectional RNN model training
rnn_b.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.10)
rnn_b.save(model_path + '\mybidirectRNN_bcrossentropy')

# rnn_b = kr.load_model(model_path + '/mybidirectRNN_bcrossentropy')


##########################################Evaluation

predictions = rnn_b.predict(x=x_test)
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(results_path)
results.save_confusion_matrix(results_path)
results.save_per_class_metrics(results_path)
results.save_macro_averaged_metrics(results_path)

# Save the best hyperparameters
with open(results_path + '/' + 'bidirect_rnn_tuned_hps.txt', 'w') as file:
    file.write('embedding_dimension: {:.5f}\n'.format(tuned_hp.get('embedding_dimension')))
    file.write('rnn_units: {:.5f}\n'.format(tuned_hp.get('rnn_units')))
    file.write('dropout_rate1: {:.5f}\n'.format(tuned_hp.get('dropout_rate1')))
    file.write('dropout_rate2: {:.5f}\n'.format(tuned_hp.get('dropout_rate2')))
    file.write('learning_rate: {:.5f}\n'.format(tuned_hp.get('learning_rate')))
    file.close()
