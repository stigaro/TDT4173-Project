import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from Source.Data_Manipulation import Loader
from Source.Feature_Extraction import Extractor

train_x, train_y, test_x, test_y = Loader.load_simple_sentence_dataset()

# Extract the maximum possible sentence length that can be found in the dataset
list_of_all_sentences = train_x.tolist() + test_x.tolist()
maximum_sentence_length = Extractor.get_max_length_from_list_of_string(list_of_all_sentences)

# Tokenize the sentences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(list_of_all_sentences)
train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

# Adds padding to tokenized sentences not at max length
train_x = pad_sequences(train_x, maxlen=maximum_sentence_length)
test_x = pad_sequences(test_x, maxlen=maximum_sentence_length)

# Reshape labels to work with tensorflow
train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(10000, 16, input_length=maximum_sentence_length))
model.add(tf.keras.layers.GRU(64, dropout=0.2))
model.add(tf.keras.layers.Dense(5, activation='sigmoid'))
model.summary()

# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = "./Models/GRU/Normal/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(train_x, train_y, epochs=10, batch_size=10, callbacks=[cp_callback], validation_split=0.2)
