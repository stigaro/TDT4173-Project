import time
from functools import wraps
import numpy as np


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
    """Turns each input to a function into lists of these inputs if not already a  list"""

    @wraps
    def wrapper(*args, **kwargs):
        return func(
            *[[a] if not type(a) == list else a for a in args],
            **{k: [v] if not type(v) == list else v for k, v in kwargs.items()}
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
        maximum_predictions[index, :] = np.identity(number_of_output_neurons)[index_of_maximum_output][:]
    return maximum_predictions


def get_max_length_from_list_of_string(string_list: list):
    return max([len(string) for string in string_list])


# Define members visible outside of module
__all__ = [k for k in globals() if k not in ['time', 'wraps']]
