from tensorflow import keras

from src.util.constants import *
from src.util.extraction import SoftmaxOutputResultsExtractor
from src.util.visualization import *

model = keras.models.load_model(PATH_TO_MODEL_GRU_SIMPLE_SAVE)
result_extractor = SoftmaxOutputResultsExtractor(model)
per_class_metrics = result_extractor.retrieve_per_class_metrics()
print_per_class_metrics(per_class_metrics)
