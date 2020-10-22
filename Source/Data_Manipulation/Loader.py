import csv
import pickle
import numpy as np


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
    with open('./Data/Preprocessed/Simple_Sentence_Dataset.pickle', 'rb') as file:
        dataset = pickle.load(file)
        train_x = np.array([x for (x, y) in dataset["train"]]).astype(np.str)
        train_y = np.array([y for (x, y) in dataset["train"]]).astype(np.int)
        test_x = np.array([x for (x, y) in dataset["test"]]).astype(np.str)
        test_y = np.array([y for (x, y) in dataset["test"]]).astype(np.int)
        return train_x, train_y, test_x, test_y


def load_simple_word_dataset():
    with open('./Data/Preprocessed/Simple_Word_Dataset.pickle', 'rb') as file:
        dataset = pickle.load(file)
        train_x = np.array([np.array(x) for (x, y) in dataset["train"]]).astype(np.ndarray)
        train_y = np.array([y for (x, y) in dataset["train"]]).astype(np.int)
        test_x = np.array([np.array(x) for (x, y) in dataset["test"]]).astype(np.ndarray)
        test_y = np.array([y for (x, y) in dataset["test"]]).astype(np.int)
        return train_x, train_y, test_x, test_y
