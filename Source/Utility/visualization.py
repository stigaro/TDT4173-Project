import matplotlib.pyplot as plt
from cycler import cycler
from tabulate import tabulate

from Source.Utility.utility import class_number_to_string_label


def generate_roc_curve_plot(per_class_roc):
    """
    Function that takes in a per-class list with FPR and TPR sequences throughout thresholding.
    From this it generates a ROC curve plot per class.
    """
    plt.figure()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'g', 'b', 'm'])))
    for class_number, (tpr, fpr) in enumerate(per_class_roc):
        plt.plot(fpr, tpr, lw=2, label=class_number_to_string_label(class_number).lower())
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def print_per_class_metrics(per_class_metrics, include_total=False):
    header = ["Metric"]
    for class_number, class_metric in enumerate(per_class_metrics):
        header.append("Class ["+str(class_number)+"]")

    row_list = []
    for index, metric_name in enumerate(list(per_class_metrics[0].keys())):
        row_list.append([metric_name])
    for index, row in enumerate(row_list):
        for class_number, class_metric in enumerate(per_class_metrics):
            row_list[index].append('{:.2f}'.format(class_metric[row_list[index][0]]))

    print(tabulate(row_list, headers=header, tablefmt='orgtbl', numalign="right", floatfmt=".2f"))
