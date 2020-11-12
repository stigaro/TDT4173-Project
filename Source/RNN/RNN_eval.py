import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def confusion_rnn(model, t_test, s_test, classes):
    # test the test data
    pred = np.argmax(model.predict(t_test), axis=-1)


    loss, acc = model.evaluate(t_test, s_test, verbose=0)
    print('Test loss: {}'.format(loss))
    print('Test Accuracy: {}'.format(acc))

    conf = confusion_matrix(s_test, pred)

    cm = pd.DataFrame(conf, index=list(classes), columns=list(classes))

    plt.figure(figsize=(12, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()



