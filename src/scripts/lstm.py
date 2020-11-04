import os

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

from src.util.extraction import WordExtractor
from src.util.loading import CSVTweetReader
from src.util import digitize

datapath = os.path.join(
    os.path.normpath(
        os.path.join(os.path.abspath(__file__), '..\..\..')    
    ),
    'Data',
    'Raw',
    'Corona_NLP_train.csv'
)

corpus = CSVTweetReader(datapath)

X = list(corpus.tokenized())
y = digitize(list(corpus.labels()))

N_FEATURES = 1000  # I.e. # of words known to the encoding
DOC_LEN = 30
N_CLASSES = len(set(y))

def build_lstm():
    lstm = Sequential()
    lstm.add(Embedding(N_FEATURES+1, 10, input_length= DOC_LEN))
    lstm.add(LSTM(units=200, activation= 'sigmoid'))
    lstm.add(Dense(N_CLASSES, activation='softmax'))
    lstm.compile(
        loss='categorical_crossentropy', 
        optimizer= Adam(learning_rate= 0.01),
        metrics=['accuracy']
    )
    return lstm

model = Pipeline([
    ('tokens', WordExtractor(nfeatures=N_FEATURES,
                             doclen=DOC_LEN)),
    ('nn', KerasClassifier(build_fn=build_lstm,
                           epochs=10,
                           batch_size=128))
])

result = cross_validate(model, X, y, cv= 5,
                        scoring= ['accuracy'])

# Print the results
for k, v in result.items():
    print(f'{k}: {v.mean()}')