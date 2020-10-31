from tensorflow import keras

from Source.Utility.constants import *
from Source.Utility.extraction import ResultsExtractor
from Source.Utility.visualization import generate_roc_curve_plot

model = keras.models.load_model(PATH_TO_MODEL_GRU_SIMPLE_SAVE)
results = ResultsExtractor(model)
tpr, fpr = results.retrieve_fpr_and_tpr_over_all_thresholds()
generate_roc_curve_plot(fpr, tpr)
