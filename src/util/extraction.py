import numpy as np
from unicodedata import category as unicat
from nltk.probability import FreqDist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import multilabel_confusion_matrix
from keras.preprocessing import sequence
from tensorflow.keras import activations

from src.util.loading import load_simple_sentence_dataset
from src.util import softmax_output_to_list_label_by_maximum


class WordExtractor(BaseEstimator, TransformerMixin):
    """
	Extract tokens, and transform
	documents into lists of these (encoded) tokens, with a total
	token lexicon limited by the nfeatures parameter
	and a document length limited/padded to doclen
	"""

    def __init__(self, nfeatures=100000, doclen=60):
        self.nfeatures = nfeatures
        self.doclen = doclen
        self.lexicon = None

    def normalize(self, sent):
        """
		Removes punctuation from a tokenized/tagged sentence and
		lowercases words.
		"""
        is_punct = lambda word: all(unicat(c).startswith('P') for c in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: t[0].lower(), sent)
        return list(sent)

    def extract_words(self, sents):
        for sent in sents:
            for word in self.normalize(sent):
                yield word

    def fit(self, documents, y=None):
        docs = [list(self.extract_words(doc)) for doc in documents]
        self.lexicon = self.get_lexicon(docs)
        return self

    def get_lexicon(self, norm_docs):
        """
		Build a lexicon of size nfeatures
		"""
        tokens = [token for doc in norm_docs for token in doc]
        fdist = FreqDist(tokens)
        counts = fdist.most_common(self.nfeatures)
        lexicon = [token for token, count in counts]
        return {token: idx + 1 for idx, token in enumerate(lexicon)}

    def clip(self, norm_doc):
        """
		Remove tokens from documents that aren't in the lexicon
		"""
        return [self.lexicon[token] for token in norm_doc
                if token in self.lexicon.keys()]

    def transform(self, documents):
        docs = [list(self.extract_words(doc)) for doc in documents]
        clipped = [list(self.clip(doc)) for doc in docs]
        return sequence.pad_sequences(clipped, maxlen=self.doclen)


class SigmoidOutputResultsExtractor:
    """
	Class that is initialized with a model which it uses to predict on the test data.
	It can then be used to extract result measurements through accessing variables or running functions.
	SHOULD ONLY BE USED WHEN THE LAST LAYER IS SIGMOID OUTPUT!
	"""
    model = None
    test_x: np.ndarray = None
    test_y: np.ndarray = None
    predictions: np.ndarray = None

    def __init__(self, model_to_use):
        """
        Initialization function to generate state containing dataset and model predictions
        """
        # Safeguard against wrong output activations
        if model_to_use.layers[-1].activation is not activations.sigmoid:
            raise Exception

        # Saves model in state, and loads dataset
        self.model = model_to_use
        ___, ___, self.test_x, self.test_y = load_simple_sentence_dataset()

        # Extracts predictions on the test dataset
        self.predictions = self.model.predict(self.test_x)

    def retrieve_per_class_roc(self):
        """
        Function that retrieves a per-class list of touples containing TPR and FPR arrays
        with values through all possible thresholds ranging from 0 -> 1.
        The returned arrays are ordered as the first index
        being a threshold of 0 and last being a threshold of 1.
        """
        thresholds = np.linspace(0, 1, 100)
        empty_array = np.zeros(100, dtype=np.float32)
        class_list_of_tpr_and_fpr = [
            (empty_array.copy(), empty_array.copy()),
            (empty_array.copy(), empty_array.copy()),
            (empty_array.copy(), empty_array.copy()),
            (empty_array.copy(), empty_array.copy()),
            (empty_array.copy(), empty_array.copy())
        ]

        for threshold_index, threshold in enumerate(thresholds):
            thresholded_predictions = np.array(self.predictions)
            thresholded_predictions[thresholded_predictions >= threshold] = 1.0
            thresholded_predictions[thresholded_predictions < threshold] = 0.0

            confusion_matrixes = multilabel_confusion_matrix(self.test_y, thresholded_predictions)
            for class_number, class_confusion_matrix in enumerate(confusion_matrixes):
                tn = class_confusion_matrix[0, 0]
                fn = class_confusion_matrix[1, 0]
                tp = class_confusion_matrix[1, 1]
                fp = class_confusion_matrix[0, 1]

                class_list_of_tpr_and_fpr[class_number][0][threshold_index] = tp / (tp + fn)
                class_list_of_tpr_and_fpr[class_number][1][threshold_index] = fp / (fp + tn)

        return class_list_of_tpr_and_fpr


class SoftmaxOutputResultsExtractor:
    """
	Class that is initialized with a model which it uses to predict on the test data.
	It can then be used to extract result measurements through accessing variables or running functions.
	SHOULD ONLY BE USED WHEN THE LAST LAYER IS SOFTMAX OUTPUT!
	"""
    model = None
    test_x: np.ndarray = None
    test_y: np.ndarray = None
    predictions: np.ndarray = None

    def __init__(self, model_to_use):
        """
        Initialization function to generate state containing dataset and model predictions
        """
        # Safeguard against wrong output activations
        if model_to_use.layers[-1].activation is not activations.softmax:
            raise Exception

        # Saves model in state, and loads dataset
        self.model = model_to_use
        ___, ___, self.test_x, self.test_y = load_simple_sentence_dataset()

        # Extracts predictions on the test dataset, and converts it to binary list label (1 at maximum output, 0 else)
        self.predictions = softmax_output_to_list_label_by_maximum(self.model.predict(self.test_x))

    def retrieve_per_class_metrics(self):
        """
        Function that retrieves a per-class list of metrics
        """
        per_class_metrics = []
        confusion_matrixes = multilabel_confusion_matrix(self.test_y, self.predictions)
        for class_number, class_confusion_matrix in enumerate(confusion_matrixes):
            tn = class_confusion_matrix[0, 0]
            fn = class_confusion_matrix[1, 0]
            tp = class_confusion_matrix[1, 1]
            fp = class_confusion_matrix[0, 1]

            p = tp + fn
            n = tn + fp

            per_class_metrics.append({
                "p": p,
                "n": n,
                "tp": tp,
                "tn": tn,
                "fn": fn,
                "fp": fp,
                "sensitivity": tp / p,
                "specificity": tn / n,
                "precision": tp / (tp + fp),
                "accuracy": (tp + tn) / (p + n),
                "balanced-accuracy": 0.5 * ((tp / p) + (tn / n)),
                "f1-score": ((2 * tp) / ((2 * tp) + fp + fn)),
                "mcc": ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            })

        return per_class_metrics

    def retrieve_total_metrics(self):
        """
        Function that retrieves a total metrics
        """
        total_metrics = {}
        p = n = tn = fn = tp = fp = 0
        confusion_matrixes = multilabel_confusion_matrix(self.test_y, self.predictions)
        for class_number, class_confusion_matrix in enumerate(confusion_matrixes):
            tn += class_confusion_matrix[0, 0]
            fn += class_confusion_matrix[1, 0]
            tp += class_confusion_matrix[1, 1]
            fp += class_confusion_matrix[0, 1]
            p += tp + fn
            n += tn + fp

        total_metrics = {
            "p": p,
            "n": n,
            "tp": tp,
            "tn": tn,
            "fn": fn,
            "fp": fp,
            "sensitivity": tp / p,
            "specificity": tn / n,
            "precision": tp / (tp + fp),
            "accuracy": (tp + tn) / (p + n),
            "balanced-accuracy": 0.5 * ((tp / p) + (tn / n)),
            "f1-score": ((2 * tp) / ((2 * tp) + fp + fn)),
            "mcc": ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        }

        return total_metrics
