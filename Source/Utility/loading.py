import os
import csv
import numpy as np
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from Source.Utility.utility import string_label_to_list_label
from Source.Utility.constants import *


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


def load_simple_sentence_dataset(tokenize=True):
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

    if tokenize is True:
        # Tokenize the sentences
        tokenizer = Tokenizer(num_words=10000)
        list_of_all_sentences = train_x.tolist() + test_x.tolist()
        tokenizer.fit_on_texts(list_of_all_sentences)
        train_x = tokenizer.texts_to_sequences(train_x)
        test_x = tokenizer.texts_to_sequences(test_x)

        # Adds padding to tokenized sentences not at max length
        train_x = pad_sequences(train_x, maxlen=MAXIMUM_SENTENCE_LENGTH)
        test_x = pad_sequences(test_x, maxlen=MAXIMUM_SENTENCE_LENGTH)

    # Reshape labels to work with tensorflow
    train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
    test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

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
    def __init__(self, path):
        """
        Reades all csv files under path and provides tokenisation methods.
        """

        print(f'Instantiating with path: {path}')

        self.root = path

        # Get all filepaths in dir if path is a dir
        self.paths = [
            os.path.join(os.path.abspath(dirpath), filename)
            for dirpath, dirnames, filenames in os.walk(self.root)
            for filename in filenames if os.path.splitext(filename)[1] == '.csv'
        ] if os.path.isdir(path) else [path]  # use filter function instead

        print(f'csv files found: {list(map(os.path.basename, self.paths))}')

        assert len(self.paths) > 0, "The provided directory/file contained no csv files"

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
        # Return everything under root if nothing is specified
        if not dir_or_filename:
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

    def labels(self, fileids=None):
        """Return the labels"""
        cleaner = self.prepare(fileids, 'id')
        for data in self.read(cleaner=cleaner):
            yield data['Sentiment']

    def tokenized(self, fileid=None):
        """
        Segments, tokenizes, and tags a text in the corpus. Returns a
        generator of texts, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        # TODO: Provide functionality for different tokenisation methods
        # and save these to files (time consuming)
        print('Tokenizing each tweet. This might take some time...')
        for text in self.texts(fileids=fileid):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(text)
            ]
