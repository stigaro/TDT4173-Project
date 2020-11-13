from src.util.loading import CSVTweetReader
from src.util.vectorizers import WordLexicolizer
from src.util.normalizers import TweetNormalizer
from src.util.tokenizers import nltk_sent_tweet_tokenizer
import src.util.constants as CONST
from src.util import label_to_int, string_label_to_list_label

import numpy as np
from sklearn.pipeline import Pipeline
"""
Various utility functions for loading in datasets,
both raw, tokenised and fully preprocessed.
"""


def get_train_reader():
    return CSVTweetReader(
        input_path= CONST.PATH_TO_RAW_TRAIN_DATA,
        output_path= CONST.CLEAN_DATA_PATH
    )

def get_test_reader():
    return CSVTweetReader(
        input_path= CONST.PATH_TO_RAW_TEST_DATA,
        output_path= CONST.CLEAN_DATA_PATH
    )

def get_all_tweets_reader():
    return CSVTweetReader(
        input_path= CONST.DATA_PATH,
        output_path= CONST.CLEAN_DATA_PATH
    )

def get_train_test_sets():
    train = get_train_reader()
    test = get_test_reader()

    train_tknzd_txts = train.tokenized(tknzr= nltk_sent_tweet_tokenizer)
    test_tknzd_txts = test.tokenized(tknzr= nltk_sent_tweet_tokenizer)

    norm_and_vectorize = Pipeline([
        ('norm', TweetNormalizer()),
        ('vect', WordLexicolizer(nfeatures= CONST.NUMBER_OF_WORDS,
                                 doclen= CONST.MAXIMUM_SENTENCE_LENGTH))
    ])

    train_x = norm_and_vectorize.fit_transform(train_tknzd_txts)
    train_y = np.array([string_label_to_list_label(l) for l in train.labels()])
    test_x = norm_and_vectorize.transform(test_tknzd_txts)  # NOTE: Use transform and NOT fit_transform here
    test_y = np.array([string_label_to_list_label(l) for l in test.labels()])
    
    print(train_x)
    print(train_y)

    return train_x, train_y, test_x, test_y