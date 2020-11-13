from sklearn.base import TransformerMixin, BaseEstimator
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
from pycontractions import Contractions
import nltk.corpus.reader.wordnet as wn

import re, nltk
from unicodedata import category as unicat

from src.util import timeit

# Download data needed by nltk utilities
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

seg_tw = Segmenter(corpus= 'twitter')
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
cont = Contractions(api_key= "glove-twitter-100")

# Used by lemmatizer to determine lemma
tags = {
    'N': wn.NOUN,
    'V': wn.VERB,
    'R': wn.ADV,
    'J': wn.ADJ
}

# NOTE: All normalisation methods expect text to a list of
# lists of (word, pos_tag) pairs (tuples).
# This formar is the result of the nltk_sent_tweet_tokenizer
# method in the src.modeling.tokenizers module.

def remove_punct_words(text):
    return [
        (token, pt) for token, pt in text
        if not all(unicat(c).startswith('P') for c in token)
    ]

def stemmer(text):
    """
    Uses nltk's snowballstemmer to stem english words
    """
    return [
        (stemmer.stem(str(word)), pt) for word, pt in text
    ]

def lower(text):
    return [
       ( token.lower(), pt) for token, pt in text
    ]

def remove_stop_words(text):
    return [
        (token, pt) for token, pt in text
        if not token in stop_words
    ]

def split_hashtags(text):
    """
    If text contains hashtags, split into a well formed phrases.
    E.g [#imsocool] -> [#, im, so, cool]
    """
    new_text = []
    for token, pt in text:
        if '#' in token:
            split = ''.join(seg_tw.segment(token))
            split = ['#'] + re.split('[^a-zA-Z0-9]', split)
            for t in split:
                new_text.append((t, pt))
        else:
            new_text.append((token, pt))
    return new_text

def remove_links(text):
    return [
        (token, pt) for token, pt in text
        if re.search('https?://\S+|www\.\S+', token) is None
    ]

def tweet_preprocess(text):
    """Preprocess tokens according to tweet-preprocessor"""
    return [
        (p.clean(token), pt) for token, pt in text
    ]

def lemmatize(text):
    return [
        (lemmatizer.lemmatize(token, tags.get(pt[0], wn.NOUN)), pt)
        for token, pt in text
    ]

def expand_contr(text):  # Extremely slow unfortunatly
    new_text = []
    for token, pt in text:
        if "'" in token:
            for exp in cont.expand_texts([token]):
                for t in exp.split(' '):
                    new_text.append((t, pt))
        else:
            new_text.append((token, pt))
    return new_text


def regex_clean(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

class TweetNormalizer(TransformerMixin, BaseEstimator):
    """
    Normamizes tweets that are expected to be sent-tokenized,
    word-tokenized and pos-tagged.
    Default normalizers are:
        - lowercasing
        - remoce links
        - remove puncuations
        - expanded hashtags
        - lemmatisation
    These can be ovrrided by providing a list of normalisation
    callbacks. See the src.modeling.normalizers module for examples on
    these methods.
    """
    def __init__(self, normalizers= None):

        # Use defaults if not specified
        self.normalizers = normalizers if normalizers is not None else [
            lower,
            remove_links,
            remove_punct_words,
            split_hashtags,
            #expand_contr,
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
        return [
            list(self.normalize(doc)) for doc in documents
        ]
    

if __name__ == "__main__":
    from src.util.constants import (CLEAN_DATA_PATH,
                                    PATH_TO_RAW_TRAIN_DATA)

    from src.util.loading import CSVTweetReader
    from src.modeling.tokenizers import nltk_sent_tweet_tokenizer

    reader = CSVTweetReader(input_path= PATH_TO_RAW_TRAIN_DATA,
                            output_path= CLEAN_DATA_PATH)

    transformer = TweetNormalizer()

    processed = reader.tokenized(tknzr= nltk_sent_tweet_tokenizer)

    transformed = transformer.transform(processed)

    for i, t in enumerate(transformed):
        if i >= 10: break
        print(i, ':')
        print(list(t))

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from nltk import FreqDist

    #Frequency of words
    fdist = FreqDist(token for doc in transformed for token in doc)
    #WordCloud
    wc = WordCloud(width=800, height=400, max_words=50).generate_from_frequencies(fdist)
    plt.figure(figsize=(12,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
