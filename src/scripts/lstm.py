import os

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, \
                            confusion_matrix, plot_confusion_matrix, \
                            f1_score, roc_curve


from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

from src.util.extraction import WordLexicolizer
from src.util.loading import CSVTweetReader
from src.util import digitize, random_sample
from src.util.constants import *

from matplotlib import pyplot as plt

import nltk

corpus = CSVTweetReader(input_path= PATH_TO_RAW_TRAIN_DATA,
                        output_path= CLEAN_DATA_PATH)

X = list(corpus.tokenized(tknzr= 'nltk_tweet'))
y = list(corpus.labels(digitized= True))

N_FEATURES = 8000  # I.e. # of words known to the encoding
DOC_LEN = 50
N_CLASSES = len(corpus.unique_labels)

def build_lstm():
    lstm = Sequential()
    lstm.add(Embedding(N_FEATURES+1, 20, input_length= DOC_LEN))
    lstm.add(Dropout(0.5))
    lstm.add(LSTM(units=100, activation= 'sigmoid'))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(N_CLASSES, activation='softmax'))
    lstm.compile(
        loss='categorical_crossentropy', 
        optimizer= Adam(learning_rate= 0.01),
        metrics=['accuracy']
    )
    lstm.summary()
    return lstm

model = Pipeline([
    ('tokens', WordLexicolizer(nfeatures=N_FEATURES,
                             doclen=DOC_LEN,
                             normalizers=[])),
    ('nn', KerasClassifier(build_fn=build_lstm,
                           epochs=10,
                           batch_size=128))
])

def cross_val():
    result = cross_validate(model, X, y, cv= 10, scoring= ['accuracy'])

    # Print the results
    for k, v in result.items():
        print(f'{k}: {v.mean()}')
        
def train_and_eval():
    global y
    #X, y = random_sample(X, y, 0.25)
    
    x_train, x_val, y_train, y_val= train_test_split(X, y,
                                                    test_size= 0.05,
                                                    shuffle= True)
    
    model.fit(x_train, y_train)
    
    preds= model.predict(x_val)
    probas= model.predict_proba(x_val)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average= 'micro')
    roc_auc = roc_auc_score(y_val, probas, average= 'weighted', multi_class= 'ovr')
    print(f'acc: {acc} | f1-micro: {f1} | roc auc: {roc_auc}')
    
    plot_confusion_matrix(model, x_val, y_val,
                          display_labels= list(corpus.unique_labels.keys()),
                          labels= list(corpus.unique_labels.values()))
    
    plt.show()
    
train_and_eval()