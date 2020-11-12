from sklearn.base import TransformerMixin, BaseEstimator
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter

import re, nltk
from unicodedata import category as unicat

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

seg_tw = Segmenter(corpus= 'twitter')
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def is_punct(word):
    return all(unicat(c).startswith('P') for c in word)

def remove_punct_words(text):
    for token, pt in text:
        if not is_punct(token):
            yield token, pt

def stemmer(text):
    """
    Uses nltk's snowballstemmer to stem english words
    """
    for word, pt in text:
        yield stemmer.stem(str(word)), pt

def lower(text):
    for token, pt in text:
        yield token.lower(), pt

def remove_stop_words(text):
    for token, pt in text:
        if not token in stop_words:
            yield token, pt

def split_hashtags(text):
    """
    If text contains hashtags, split into a well formed phrases.
    E.g [#imsocool] -> [#, im, so, cool]
    """
    for token, pt in text:
        if '#' in token:
            for t in seg_tw.segment(token):
                yield t, pt
        else:
            yield token, pt

def remove_links(text):
    for token, pt in text:
        if re.search('https?://\S+|www\.\S+', token) is None:
            yield token, pt

def tweet_preprocess(text):
    """Preprocess tokens according to tweet-preprocessor"""
    for token, pt in text:
        yield p.clean(token), pt

def lemmatize(text):
    import nltk.corpus.reader.wordnet as wn

    tags = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }

    for token, pt in text:
        tag = tags.get(pt[0], wn.NOUN)
        yield lemmatizer.lemmatize(token, tag), pt

def regex_clean(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

class TweetNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, normalizers= None):
        
        # Use defaults if not specified
        self.normalizers = normalizers if normalizers is not None else [
            lower,
            remove_punct_words,
            lemmatize
        ]
    
    def fit(self, X, y= None):
        return self
    
    def normalize(self, doc):
        for sent in doc:
            # Normalize with every normalizer
            for norm in self.normalizers:
                sent = norm(sent)
            
            # Yield evey token from normalized sent
            for token, pt in sent:
                if token:
                    yield token
    
    def transform(self, documents):
        return [list(self.normalize(doc)) for doc in documents]
    

if __name__ == "__main__":
    from src.util.constants import (CLEAN_DATA_PATH,
                                    PATH_TO_RAW_TRAIN_DATA)

    from src.util.loading import CSVTweetReader

    reader = CSVTweetReader(input_path= PATH_TO_RAW_TRAIN_DATA,
                            output_path= CLEAN_DATA_PATH)

    transformer = TweetNormalizer()

    processed = reader.tokenized(tknzr= 'nltk_sent_tweet')

    transformed = transformer.transform(processed)

    for i, t in enumerate(transformed):
        if i >= 10: break
        print(i, ':')
        print(t)




"""    
import pandas as pd
import numpy as np
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist

#Frequency of words
fdist = FreqDist(tweets['Segmented#'])
#WordCloud
wc = WordCloud(width=800, height=400, max_words=50).generate_from_frequencies(fdist)
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
"""