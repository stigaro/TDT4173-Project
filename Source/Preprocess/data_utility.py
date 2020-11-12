
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
import string


# data path (to be modified)
data_path = r".\Resources\Data"


def read_data(): # read train and test data
    train = pd.read_csv(data_path + "\Corona_NLP_train.csv", encoding='latin1')
    test = pd.read_csv(data_path + "\Corona_NLP_test.csv", encoding='latin1')
    df = train.append(test, sort = False)
    return df


def clean_data(t):
    t = str(t).lower()  # lowercase
    t = re.sub('\[.*?\]', '', t)  # brackets
    t = re.sub('https?://\S+|www\.\S+', '', t)  # URLs
    t = re.sub('<.*?>+', '', t)  # punctuation
    t = re.sub('[%s]' % re.escape(string.punctuation), '', t)  # punctuation
    t = re.sub('\n', '', t)  # punctuation
    t = re.sub('\w*\d\w*', '', t)  # punctuation
    return t


def token(t):
    # tokenize the tweets
    tok = Tokenizer()
    tok.fit_on_texts(t)
    tweet = tok.texts_to_sequences(t)

    vocab_size = len(tok.word_index) + 1



    return tweet, vocab_size


def padd(t):
    t = pad_sequences(t, padding='post')
    return t


def enco(df):
    encoder = LabelEncoder()
    df['encoded_sentiment'] = encoder.fit_transform(df['Sentiment'])
    sen = df['encoded_sentiment']
    return sen, encoder