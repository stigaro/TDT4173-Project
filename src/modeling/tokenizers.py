import nltk
from nltk import word_tokenize, wordpunct_tokenize, \
    TweetTokenizer, pos_tag, sent_tokenize

nltk.download('averaged_perceptron_tagger')

tknze = TweetTokenizer(reduce_len=True, strip_handles=True).tokenize

def nltk_tweet_tokenizer(text):
    tknzr = TweetTokenizer(reduce_len= True, strip_handles= True)
    return tknzr.tokenize(text)

def nltk_wordpunct_tokenizer(text):
    return wordpunct_tokenize(text)

def nltk_sent_tweet_tokenizer(text):
    return [
        pos_tag(tknze(sent))
        for sent in sent_tokenize(text)
    ]

_EXCLUDE = {'word_tokenize', 'wordpunct_tokenize', 'TweetTokenizer'}

__all__ = [k for k in globals() if k not in _EXCLUDE and not k.startswith('_')]