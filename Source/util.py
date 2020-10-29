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
    d = dict([(x, i+1) for i, x in enumerate(sorted(set(labels)))])
    return np.array([d[x] for x in labels])

# Define members visible outside of module
__all__ = [k for k in globals() if k not in ['time', 'wraps']]