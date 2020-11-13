from keras.preprocessing import sequence
from nltk.probability import FreqDist
from sklearn.base import BaseEstimator, TransformerMixin

class WordLexicolizer(BaseEstimator, TransformerMixin):
    """
	Encodes tokens in tokenized documents, with a total
	token-to-encoding lexicon limited by the nfeatures parameter
	and a document length limited/padded to doclen
	"""

    def __init__(self, nfeatures=100000, doclen=60):
        self.nfeatures = nfeatures
        self.doclen = doclen
        self.lexicon = None

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
        return [
            self.lexicon[token]
            if token in self.lexicon.keys() else 0  # Unknown reserved as 0
            for token in norm_doc
        ]

    def fit(self, documents, y=None):
        self.lexicon = self.get_lexicon(documents)
        print('The most common word according to the encoding is: ')
        print([t[0] for t in sorted(self.lexicon.items(), key= lambda i: i[1])][:100])
        return self

    def transform(self, documents):
        clipped = [list(self.clip(doc)) for doc in documents]
        return sequence.pad_sequences(clipped, maxlen=self.doclen)