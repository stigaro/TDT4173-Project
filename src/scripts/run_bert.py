import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

from src.util.constants import *
from src.util.extraction import ResultsExtractor
from src.util.loading import load_simple_sentence_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Load the dataset
train_x, train_y, test_x, test_y = load_simple_sentence_dataset()

# Tokenize the sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_x = tokenizer(train_x.tolist(), padding='max_length', return_tensors='tf', max_length=MAXIMUM_SENTENCE_LENGTH)
test_x = tokenizer(test_x.tolist(), padding='max_length', return_tensors='tf', max_length=MAXIMUM_SENTENCE_LENGTH)

# Reshape labels to work with tensorflow
train_y = np.asarray(train_y).astype('float32').reshape((-1, 5))
test_y = np.asarray(test_y).astype('float32').reshape((-1, 5))

# Loads pre trained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)

# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = PATH_TO_MODEL_TRANSFORMER_BERT_CHECKPOINTS + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=1
)

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), metrics=['accuracy'])

# Fit the model and save it
model.fit(x=train_x.values(), y=train_y, epochs=5, batch_size=8, callbacks=[cp_callback], validation_split=0.10)
model.save_pretrained(PATH_TO_MODEL_TRANSFORMER_BERT_SAVE)

# Save the results
predictions = model.predict(x=test_x.values())[0]
results = ResultsExtractor(predictions)
results.save_per_class_roc_curves(PATH_TO_RESULT_TRANSFORMER_BERT)
results.save_confusion_matrix(PATH_TO_RESULT_TRANSFORMER_BERT)
results.save_per_class_metrics(PATH_TO_RESULT_TRANSFORMER_BERT)
results.save_macro_averaged_metrics(PATH_TO_RESULT_TRANSFORMER_BERT)
