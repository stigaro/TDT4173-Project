import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from Source.Utility.constants import *
from Source.Utility.extraction import ResultsExtractor
from Source.Utility.generation import Generator
from Source.Utility.loading import load_simple_sentence_dataset
from Source.Utility.visualization import save_per_class_metrics, save_per_class_roc_curves

# Load the dataset
train_x, train_y, test_x, test_y = load_simple_sentence_dataset()

# Tokenize the sentences
tokenizer = Tokenizer(num_words=10000, oov_token=True)
tokenizer.fit_on_texts(train_x.tolist())
train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

# Adds padding to tokenized sentences not at max length
train_x = pad_sequences(train_x, maxlen=MAXIMUM_SENTENCE_LENGTH)
test_x = pad_sequences(test_x, maxlen=MAXIMUM_SENTENCE_LENGTH)

# Reshape labels to work with tensorflow
train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

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
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

# Fit the model and save it
model.fit(train_x, train_y, epochs=5, batch_size=10, callbacks=[cp_callback], validation_split=0.10)
model.save(PATH_TO_MODEL_GRU_BIDIRECTIONAL_SAVE)

# Save the results
predictions = model.predict(x=test_x)
results = ResultsExtractor(predictions)
save_per_class_metrics(results.retrieve_per_class_metrics(), PATH_TO_RESULT_GRU_BIDIRECTIONAL)
save_per_class_roc_curves(results.retrieve_per_class_roc(), PATH_TO_RESULT_GRU_BIDIRECTIONAL)
