import matplotlib.pyplot as plt


def generate_roc_curve_plot(fpr, tpr):
    """
    Function that takes in a FPR and TPR sequence through thresholding, and generates a ROC curve plot
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
