import os
import csv
import pickle
from sys import path
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from pycontractions import Contractions

from src.util import string_label_to_list_label, listemize_input, timeit
from src.util.constants import *
from src.util.tokenizers import *


def load_raw_training_data():
    raw_data = []
    with open(PATH_TO_RAW_TRAIN_DATA, encoding='latin-1') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skips metadata
        for row in reader:
            raw_data.append(row)
    return raw_data


def load_raw_testing_data():
    raw_data = []
    with open(PATH_TO_RAW_TEST_DATA, encoding='latin-1') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skips metadata
        for row in reader:
            raw_data.append(row)
    return raw_data


def load_simple_sentence_dataset():
    training_data = load_raw_training_data()
    testing_data = load_raw_testing_data()

    training_x = np.array([np.array(row[4]) for row in training_data])
    training_y = np.array([string_label_to_list_label(row[5]) for row in training_data])

    testing_x = np.array([np.array(row[4]) for row in testing_data])
    testing_y = np.array([string_label_to_list_label(row[5]) for row in testing_data])

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
    training_y = np.array([string_label_to_list_label(row[5]) for row in training_data])

    testing_x = np.array([np.array(row[4].split()) for row in testing_data])
    testing_y = np.array([string_label_to_list_label(row[5]) for row in testing_data])

    dataset = {
        "train": np.array([(x, y) for x, y in zip(training_x, training_y)]),
        "test": np.array([(x, y) for x, y in zip(testing_x, testing_y)])
    }

    train_x = np.array([np.array(x) for (x, y) in dataset["train"]]).astype(np.ndarray)
    train_y = np.array([y for (x, y) in dataset["train"]]).astype(np.int)
    test_x = np.array([np.array(x) for (x, y) in dataset["test"]]).astype(np.ndarray)
    test_y = np.array([y for (x, y) in dataset["test"]]).astype(np.int)

    # Reshape labels to work with tensorflow
    train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
    test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

    return train_x, train_y, test_x, test_y


class CSVTweetReader(object):
    def __init__(self, input_path, output_path):
        """
        Reades all csv files under input_path and provides tokenisation methods
        which results are saved to output_path.
        """

        self.root = input_path
        self.output_path = output_path

        # Get all filepaths in dir if path is a dir
        self.paths = [
            os.path.join(os.path.abspath(dirpath), filename)
            for dirpath, dirnames, filenames in os.walk(self.root)
            for filename in filenames if os.path.splitext(filename)[1] == '.csv'
        ] if os.path.isdir(input_path) else [input_path]

        self.csv_files = list(map(os.path.basename, self.paths))
        self._unique_labels = None

        print(f'Instantiating with path: {input_path}')
        print(f'csv files found: {self.csv_files}')
        print()
        print(f'Processed data will be outputted to {output_path}')

        assert len(self.paths) > 0, \
            "The provided directory/file contained no csv files"

    def read(self, reader=csv.DictReader, cleaner=lambda r: r, dir_or_filename=None):
        """
        Yields processed/cleaned datapoints from the csv files under
        dir_or_filenmae.
        
            Args:
                reader (func): callback that accepts a file object and returns a
                itereable of the rows in the file

                cleaner (func): callback that accepts individual rows from reader
                and returns a processed/cleaned datastructure from it.
        """
        if dir_or_filename is None:
            dir_or_filename = os.path.dirname(self.root)

        for path in self.paths:
            if dir_or_filename in path:  # Only return contents of relevant files
                with open(path, encoding="ISO-8859-1") as csvfile:
                    for i, row in enumerate(reader(csvfile)):
                        try:
                            row['id'] = i
                            yield cleaner(row)
                        except Exception as e:
                            pass

    def prepare(self, item, field):
        """Return a reader function based on inputs"""
        if not item:
            def cleaner(data):
                return data
        else:
            if not type(item) == list: item = [item]

            def cleaner(data):
                assert data[field] in set(item)
                return data
        return cleaner

    def texts(self, fileids=None):
        """
        Returns the unprocessed tweet texts
        """
        cleaner = self.prepare(fileids, 'id')
        for data in self.read(cleaner=cleaner):
            yield data['OriginalTweet']

    def fileids(self, categories=None):
        """Returns the tweet ids"""
        cleaner = self.prepare(categories, "Sentiment")
        for data in self.read(cleaner=cleaner):
            yield data['id']

    @property
    def unique_labels(self):
        """Return the unique labels from the initialized dataset"""
        if self._unique_labels is None:
            self._unique_labels = dict(
                (x, i + 1) for i, x in enumerate(sorted(set(self.labels())))
            )
        return self._unique_labels

    def labels(self, fileids=None, digitized=False):
        """Return the labels for the fileids"""
        cleaner = self.prepare(fileids, 'id')
        apply = (lambda sent: self.unique_labels[sent]) if digitized else (lambda x: x)

        for data in self.read(cleaner=cleaner):
            yield apply(data['Sentiment'])

    def tokenize(self, fileids, tokenizer):
        """Performs the tokenization according to tokenizer"""
        return [
            tokenizer(text)
            for text in self.texts(fileids=fileids)
        ]

    def load(self, path):
        """
        Returns dict under filename if exists.
        Otherwise, return en empty dict
        """
        file_path = os.path.join(self.output_path, path)
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            return {}

    def save(self, path, tknzd):
        """Save and overwrite contents of filename"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb+') as f:  # wb+ overwrites
            pickle.dump(tknzd, f)

    @listemize_input
    def get_id(self, csvs, fileids):
        # Sort and stringify to create reproduceable key
        return str(sorted(csvs)) + str(sorted(fileids))

    @timeit
    def tokenized(self, fileids=None, tknzr=nltk_tweet_tokenizer):
        """
        Retrieves the tokenized dataset from a pickled file if exists.
        If not, tokenizes, saves, and finally returns it.
        """
        file_path = os.path.join(self.output_path,
                                 f'{tknzr.__name__[:-10]}.pickle')
        file_contents = self.load(file_path)
        idx = self.get_id(self.csv_files, fileids)

        try:
            tknzd = file_contents[idx]
            print(f'Retrieved existing tokenised dataset from \n {file_path}')

        except Exception as e:
            print(f'Tokenizing {self.csv_files} with {tknzr.__name__} for the first time.')
            print('This might take some time...')
            tknzd = list(self.tokenize(fileids, tknzr))
            file_contents[idx] = tknzd  # Update
            self.save(file_path, file_contents)
            print(f'Result saved to {file_path} under key: {idx}')

        return tknzd

    _EXCLUDE = {
        'get_id', 'save', 'load',
        'tokenize', 'get_str_from_tknzr',
        'prepare'
    }

    # __all__ = [k for k in globals() if k not in _EXCLUDE and not k.startswith('_')]


if __name__ == "__main__":
    import pandas as pd

    reader = CSVTweetReader(input_path=PATH_TO_RAW_TRAIN_DATA,
                            output_path=CLEAN_DATA_PATH)

    # Verify data reader function
    df = pd.DataFrame(data=reader.read())
    print(df)
    print()
    print(df.info())

    # Check results from tokenisation by inspection
    for i, data in enumerate(reader.texts()):
        if i >= 10: break
        print(i, ': ')
        print('Raw:')
        print(data)
        print()
        print('tokenized: ')
        print([
            tkn
            for tkn in nltk_sent_tweet_tokenizer(data)
        ])
