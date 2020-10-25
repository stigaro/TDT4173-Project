import tensorflow as tf
import tensorflow.keras.preprocessing.text as tf_text
import numpy as np

from Source.Data_Manipulation import Loader
from Source.Feature_Extraction import Extractor

train_x, train_y, test_x, test_y = Loader.load_simple_sentence_dataset()

# Extract the maximum possible sentence length that can be found in the dataset
list_of_all_sentences = train_x.tolist() + test_x.tolist()
maximum_sentence_length = Extractor.get_max_length_from_list_of_string(list_of_all_sentences)

# Tokenize the sentences
tokenizer = tf_text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(list_of_all_sentences)
vector = tokenizer.texts_to_sequences(list_of_all_sentences)
