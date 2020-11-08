import time, re, string, random, math
import nltk
from functools import wraps
from unicodedata import category as unicat
import numpy as np

def simple_cleaner(sent):
    """
	Removes punctuation from a tokenized/tagged sentence and
	lowercases words.
	"""
    is_punct = lambda word: all(unicat(c).startswith('P') for c in word)
    sent = filter(lambda t: not is_punct(t), sent)
    sent = map(lambda t: t.lower(), sent)
    return list(sent)

def atap_simple_cleaner(token):
    """
    Simplest cleaner used in the book
    Aplied Text Analysis with Python.
    
    Lowercases and stems.
    """
    stem = nltk.stem.SnowballStemmer('english')
    return stem.stem(str(token).lower())

def kaggle_regex_cleaner(doc):
    '''
    cleaner used by most read notebook on twitter competition
    
    Make text lowercase, remove text in square brackets,remove links,
    remove punctuation and remove words containing numbers.'''
    def regex_clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    
    doc = simple_cleaner(doc)
    for token in doc:
        cleaned = regex_clean(token)
        cleaned = atap_simple_cleaner(cleaned)
        if not cleaned == '' or not cleaned is None:
            yield cleaned

def timeit(func):
    """Times a function and prints it"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        delta = time.time() - start
        print('Elapsed (s): ', delta)
        return result

    return wrapper


def listemize_input(func):
    """
    Turns each input to a function into lists of these 
    inputs if not already a list
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(
            # Modify args
            *[[a] if not type(a) == list else a for a in args],
            
            # Modify kwargs
            *{
                k: [v] if not type(v) == list else v
                for k, v in kwargs.items()
            }
        )

    return wrapper


def digitize(labels):
    """Return digitized versions of the labels"""
    d = dict([(x, i + 1) for i, x in enumerate(sorted(set(labels)))])
    return np.array([d[x] for x in labels])


def list_label_to_string_label(label: list):
    return {
        [1, 0, 0, 0, 0]: 'EXTREMELY NEGATIVE',
        [0, 1, 0, 0, 0]: 'NEGATIVE',
        [0, 0, 1, 0, 0]: 'NEUTRAL',
        [0, 0, 0, 1, 0]: 'POSITIVE',
        [0, 0, 0, 0, 1]: 'EXTREMELY POSITIVE',
    }[label]


def class_number_to_string_label(class_number: int):
    return {
        0: 'EXTREMELY NEGATIVE',
        1: 'NEGATIVE',
        2: 'NEUTRAL',
        3: 'POSITIVE',
        4: 'EXTREMELY POSITIVE',
    }[class_number]


def string_label_to_list_label(label: str):
    return {
        'EXTREMELY NEGATIVE': [1, 0, 0, 0, 0],
        'NEGATIVE': [0, 1, 0, 0, 0],
        'NEUTRAL': [0, 0, 1, 0, 0],
        'POSITIVE': [0, 0, 0, 1, 0],
        'EXTREMELY POSITIVE': [0, 0, 0, 0, 1],
    }[label.upper()]


def softmax_output_to_list_label_by_maximum(predictions: np.ndarray):
    """
    Converts a softmax output (Probability for one specific class)
    to a list label by taking the class with max/most probability
    :return: The list label
    """
    maximum_predictions = np.zeros(predictions.shape)
    for index, class_predictions in enumerate(predictions):
        number_of_output_neurons = predictions.shape[1]
        index_of_maximum_output = np.argmax(class_predictions)
        maximum_predictions[index, :] =\
            np.identity(number_of_output_neurons)[index_of_maximum_output][:]
    return maximum_predictions


def get_max_length_from_list_of_string(string_list: list):
    return max([len(string) for string in string_list])

def random_sample(x, y, norm_size):
    assert len(x) == len(y), 'list x and y should have same size'
    idxs = list(range(len(x)))
    rand_idxs = [
        idxs.pop(random.randrange(0, len(idxs)))
        for x in range(math.floor(len(x) * norm_size))
    ]
    return [x[i] for i in rand_idxs], [y[i] for i in rand_idxs]

# Define members visible outside of module
_EXCLUDE = {"np", "time", "wraps"}
__all__ = [
    k for k in globals()
    if k not in _EXCLUDE and not k.startswith('_')
]
