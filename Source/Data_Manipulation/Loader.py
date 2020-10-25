import csv
import numpy as np

from Source.Feature_Extraction import Extractor


def load_raw_training_data():
    raw_data = []
    with open('./Data/Raw/Corona_NLP_train.csv', encoding='latin-1') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skips metadata
        for row in reader:
            raw_data.append(row)
    return raw_data


def load_raw_testing_data():
    raw_data = []
    with open('./Data/Raw/Corona_NLP_test.csv', encoding='latin-1') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skips metadata
        for row in reader:
            raw_data.append(row)
    return raw_data


def load_simple_sentence_dataset():
    training_data = load_raw_training_data()
    testing_data = load_raw_testing_data()

    training_x = np.array([np.array(row[4]) for row in training_data])
    training_y = np.array([Extractor.string_label_to_list_label(row[5]) for row in training_data])

    testing_x = np.array([np.array(row[4]) for row in testing_data])
    testing_y = np.array([Extractor.string_label_to_list_label(row[5]) for row in testing_data])

    dataset = {
        "train": np.array([(x, y) for x, y in zip(training_x, training_y)]),
        "test": np.array([(x, y) for x, y in zip(testing_x, testing_y)])
    }

    train_x = np.array([x for (x, y) in dataset["train"]]).astype(np.str)
    train_y = np.array([y for (x, y) in dataset["train"]]).astype(np.int)
    test_x = np.array([x for (x, y) in dataset["test"]]).astype(np.str)
    test_y = np.array([y for (x, y) in dataset["test"]]).astype(np.int)

    return train_x, train_y, test_x, test_y


def load_simple_word_dataset():
    training_data = load_raw_training_data()
    testing_data = load_raw_testing_data()

    training_x = np.array([np.array(row[4].split()) for row in training_data])
    training_y = np.array([Extractor.string_label_to_list_label(row[5]) for row in training_data])

    testing_x = np.array([np.array(row[4].split()) for row in testing_data])
    testing_y = np.array([Extractor.string_label_to_list_label(row[5]) for row in testing_data])

    dataset = {
        "train": np.array([(x, y) for x, y in zip(training_x, training_y)]),
        "test": np.array([(x, y) for x, y in zip(testing_x, testing_y)])
    }

    train_x = np.array([np.array(x) for (x, y) in dataset["train"]]).astype(np.ndarray)
    train_y = np.array([y for (x, y) in dataset["train"]]).astype(np.int)
    test_x = np.array([np.array(x) for (x, y) in dataset["test"]]).astype(np.ndarray)
    test_y = np.array([y for (x, y) in dataset["test"]]).astype(np.int)

    return train_x, train_y, test_x, test_y
