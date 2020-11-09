import numpy as np
import tensorflow as tf
import pandas as pd
from cycler import cycler
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import category as unicat
from nltk.probability import FreqDist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import multilabel_confusion_matrix
from keras.preprocessing import sequence
from tensorflow.keras import activations

from src.util.loading import load_simple_sentence_dataset
from src.util.util import softmax_output_to_list_label_by_maximum, list_label_to_number_label, \
    class_number_to_string_label

sns.set_style('whitegrid')


class WordLexicolizer(BaseEstimator, TransformerMixin):
    """
	Encodes tokens in tokenized documents, with a total
	token-to-encoding lexicon limited by the nfeatures parameter
	and a document length limited/padded to doclen
	"""

    def __init__(self, nfeatures=100000, doclen=60, normalizers=[]):
        self.nfeatures = nfeatures
        self.doclen = doclen
        self.normalizers = normalizers
        self.lexicon = None

    def normalize(self, doc):
        for norm in self.normalizers:
            doc = norm(doc)
        return doc

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
        docs = [list(self.normalize(doc)) for doc in documents]
        self.lexicon = self.get_lexicon(docs)
        print('The most common word according to the encoding is: ')
        print([t[0] for t in sorted(self.lexicon.items(), key=lambda i: i[1])][:100])
        return self

    def transform(self, documents):
        docs = [self.normalize(doc) for doc in documents]
        clipped = [list(self.clip(doc)) for doc in docs]
        return sequence.pad_sequences(clipped, maxlen=self.doclen)


class ResultsExtractor:
    """
    Class that is initialized with a models logit predictions,
    and can then be used to extract different types of result metrics
    NB: MUST ONLY BE INITIALIZED WITH LOGITS OUTPUT
    """
    test_y: np.ndarray = None
    logits: np.ndarray = None

    def __init__(self, logits):
        """
        Initialization function to generate state containing true labels and model predictions
        """
        self.logits = logits
        ___, ___, ___, self.test_y = load_simple_sentence_dataset()

    def __retrieve_per_class_roc(self):
        """
        Function that retrieves a per-class list of touples containing TPR and FPR arrays
        with values through all possible thresholds ranging from 0 -> 1.
        The returned arrays are ordered as the first index
        being a threshold of 0 and last being a threshold of 1.
        """
        predictions = np.array(tf.math.sigmoid(self.logits).numpy())

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
            thresholded_predictions = np.array(predictions)
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

    def __retrieve_per_class_metrics(self):
        """
        Function that retrieves a per-class list of metrics
        """
        predictions = np.array(tf.math.softmax(self.logits).numpy())
        predictions = softmax_output_to_list_label_by_maximum(predictions)

        per_class_metrics = []
        confusion_matrixes = multilabel_confusion_matrix(self.test_y, predictions)
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

    # noinspection PyPackageRequirements
    def __retrieve_total_metrics(self):
        """
        Function that retrieves a total metrics
        """
        predictions = np.array(tf.math.softmax(self.logits).numpy())
        predictions = softmax_output_to_list_label_by_maximum(predictions)

        p = n = tn = fn = tp = fp = 0
        confusion_matrixes = multilabel_confusion_matrix(self.test_y, predictions)
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

    def save_confusion_matrix(self, filepath: str):
        predictions = np.array(tf.math.softmax(self.logits).numpy())
        predictions = np.array(softmax_output_to_list_label_by_maximum(predictions)).astype(np.int)
        predictions = np.array([list_label_to_number_label(label) for label in predictions])
        test_y = np.array([list_label_to_number_label(label) for label in self.test_y])

        plt.figure()
        plt.title('Test')
        tick_labels = ['Extremely\nNegative', 'Negative', 'Neutral', 'Positive', 'Extremely\nPositive']
        crosstab = pd.crosstab(test_y, predictions).fillna(0).astype(int)
        crosstab.columns.name = ''
        crosstab.index.name = ''
        sns.heatmap(
            crosstab,
            annot=True,
            fmt='d',
            xticklabels=tick_labels,
            yticklabels=tick_labels
        )
        plt.yticks(va='center')
        plt.tight_layout()
        plt.savefig(filepath + '/' + 'confusion_matrix.png')

    def save_per_class_roc_curves(self, filepath: str):
        """
        Function that takes in a per-class list with FPR and TPR sequences throughout thresholding.
        From this it generates a ROC curve plot per class.
        """
        per_class_roc_curves = self.__retrieve_per_class_roc()

        plt.figure()
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'g', 'b', 'm'])))
        for class_number, (tpr, fpr) in enumerate(per_class_roc_curves):
            plt.plot(fpr, tpr, lw=2, label=class_number_to_string_label(class_number).lower())
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(filepath + '/' + 'per_class_roc_curves.png')

    def save_per_class_metrics(self, filepath: str):
        per_class_metrics = self.__retrieve_per_class_metrics()

        header = ["Metric"]
        for class_number, class_metric in enumerate(per_class_metrics):
            header.append("Class [" + class_number_to_string_label(class_number) + "]")

        row_list = []
        for index, metric_name in enumerate(list(per_class_metrics[0].keys())):
            row_list.append([metric_name])
        for index, row in enumerate(row_list):
            for class_number, class_metric in enumerate(per_class_metrics):
                row_list[index].append('{:.2f}'.format(class_metric[row_list[index][0]]))

        tabular: str = tabulate(row_list, headers=header, tablefmt='orgtbl', numalign="right", floatfmt=".2f")

        with open(filepath + '/' + 'per_class_metrics.txt', 'w') as file:
            file.write(tabular)
            file.close()

    def save_macro_averaged_metrics(self, filepath: str):
        per_class_metrics = self.__retrieve_per_class_metrics()
        header = ["Metric", "Macro-Averaging"]

        row_list = []
        for index, metric_name in enumerate(list(per_class_metrics[0].keys())):
            row_list.append([metric_name])

            macro_number = 0
            number_of_classes = len(per_class_metrics)
            for class_number in range(0, number_of_classes):
                macro_number += per_class_metrics[class_number][metric_name]
            macro_number /= number_of_classes
            row_list[index].append('{:.2f}'.format(macro_number))

        tabular: str = tabulate(row_list, headers=header, tablefmt='orgtbl', numalign="right", floatfmt=".2f")

        with open(filepath + '/' + 'macro_metrics.txt', 'w') as file:
            file.write(tabular)
            file.close()

    @staticmethod
    def save_best_hyperparameters(hyperparameters, filepath: str):
        with open(filepath + '/' + 'best_hyperparameters.txt', 'w') as file:
            file.write('embedding_output_dim: {:.5f}\n'.format(hyperparameters.get('embedding_output_dim')))
            file.write('gru_hidden_units: {:.5f}\n'.format(hyperparameters.get('gru_hidden_units')))
            file.write('gru_dropout: {:.5f}\n'.format(hyperparameters.get('gru_dropout')))
            file.write('learning_rate: {:.5f}\n'.format(hyperparameters.get('learning_rate')))
            file.close()
