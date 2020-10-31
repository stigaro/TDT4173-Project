import numpy as np
from unicodedata import category as unicat
from nltk.probability import FreqDist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import multilabel_confusion_matrix
from keras.preprocessing import sequence

from Source.Utility.loading import load_simple_sentence_dataset


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


class ResultsExtractor:
    """
	Class that is initialized with a model which it uses to predict on the test data.
	It can then be used to extract result measurements through accessing variables or running functions.
	"""
    model = None
    test_x: np.ndarray = None
    test_y: np.ndarray = None
    predictions: np.ndarray = None

    """
    Initialization function to generate state containing dataset and model predictions
    """
    def __init__(self, model_to_use):
        self.model = model_to_use
        ___, ___, self.test_x, self.test_y = load_simple_sentence_dataset()
        self.predictions = self.model.predict(self.test_x)

    """
    Function that retrieves a touple of TPR and FPR arrays with 
    values through all possible thresholds ranging from 0 -> 1. 
    The returned arrays are ordered as the first index 
    being a threshold of 0 and last being a threshold of 1
    """
    def retrieve_fpr_and_tpr_over_all_thresholds(self):
        thresholds = np.linspace(0, 1, 100)
        tpr = np.zeros(100, dtype=np.float32)
        fpr = np.zeros(100, dtype=np.float32)
        for index, threshold in enumerate(thresholds):
            thresholded_predictions = np.array(self.predictions)
            thresholded_predictions[thresholded_predictions >= threshold] = 1.0
            thresholded_predictions[thresholded_predictions < threshold] = 0.0

            tp = tn = fp = fn = 0
            confusion_matrixes = multilabel_confusion_matrix(self.test_y, thresholded_predictions)
            for class_confusion_matrix in confusion_matrixes:
                tn += class_confusion_matrix[0, 0]
                fn += class_confusion_matrix[1, 0]
                tp += class_confusion_matrix[1, 1]
                fp += class_confusion_matrix[0, 1]

            tpr[index] = tp / (tp + fn)
            fpr[index] = fp / (fp + tn)

        return tpr, fpr
