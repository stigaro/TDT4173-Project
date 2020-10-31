from tensorflow import keras

from Source.Utility.constants import *
from Source.Utility.extraction import SoftmaxOutputResultsExtractor
from Source.Utility.visualization import *

model = keras.models.load_model(PATH_TO_MODEL_GRU_SIMPLE_SAVE)
result_extractor = SoftmaxOutputResultsExtractor(model)
per_class_metrics = result_extractor.retrieve_per_class_metrics()
print_per_class_metrics(per_class_metrics)
