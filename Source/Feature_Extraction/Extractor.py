def list_label_to_string_label(label: list):
    return {
        [1, 0, 0, 0, 0]: 'EXTREMELY NEGATIVE',
        [0, 1, 0, 0, 0]: 'NEGATIVE',
        [0, 0, 1, 0, 0]: 'NEUTRAL',
        [0, 0, 0, 1, 0]: 'POSITIVE',
        [0, 0, 0, 0, 1]: 'EXTREMELY POSITIVE',
    }[label]


def string_label_to_list_label(label: str):
    return {
        'EXTREMELY NEGATIVE': [1, 0, 0, 0, 0],
        'NEGATIVE': [0, 1, 0, 0, 0],
        'NEUTRAL': [0, 0, 1, 0, 0],
        'POSITIVE': [0, 0, 0, 1, 0],
        'EXTREMELY POSITIVE': [0, 0, 0, 0, 1],
    }[label.upper()]


def get_max_length_from_list_of_string(string_list: list):
    return max([len(string) for string in string_list])



from unicodedata import category as unicat
from nltk.probability import FreqDist
from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing import sequence

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
		return {token: idx+1 for idx, token in enumerate(lexicon)}

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