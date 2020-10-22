import numpy as np
import pickle as pk

from Source.Feature_Extraction import Extractor
from Source.Data_Manipulation import Loader

training_data = Loader.load_raw_training_data()
testing_data = Loader.load_raw_testing_data()

training_x = np.array([np.array(row[4].split()) for row in training_data])
training_y = np.array([Extractor.string_label_to_integer_label(row[5]) for row in training_data])

testing_x = np.array([np.array(row[4].split()) for row in testing_data])
testing_y = np.array([Extractor.string_label_to_integer_label(row[5]) for row in testing_data])

dataset = {
    "train": np.array([(x, y) for x, y in zip(training_x, training_y)]),
    "test": np.array([(x, y) for x, y in zip(testing_x, testing_y)])
}

with open('./Data/Preprocessed/Simple_Word_Dataset.pickle', 'wb') as output:
    pk.dump(dataset, output)
