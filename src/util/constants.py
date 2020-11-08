PATH_TO_RAW_TEST_DATA: str = './Resources/Data/Raw/Corona_NLP_test.csv'
PATH_TO_RAW_TRAIN_DATA: str = './Resources/Data/Raw/Corona_NLP_train.csv'

PATH_TO_MODEL_GRU_SIMPLE_SAVE: str = './Resources/Models/GRU/Simple/Saved'
PATH_TO_MODEL_GRU_BIDIRECTIONAL_SAVE: str = './Resources/Models/GRU/Bidirectional/Saved'
PATH_TO_MODEL_TRANSFORMER_BERT_SAVE: str = './Resources/Models/Transformer/BERT/Saved'

PATH_TO_MODEL_GRU_SIMPLE_CHECKPOINTS: str = './Resources/Models/GRU/Simple/Checkpoints'
PATH_TO_MODEL_GRU_BIDIRECTIONAL_CHECKPOINTS: str = './Resources/Models/GRU/Bidirectional/Checkpoints'
PATH_TO_MODEL_TRANSFORMER_BERT_CHECKPOINTS: str = './Resources/Models/Transformer/BERT/Checkpoints'

PATH_TO_MODEL_GRU_SIMPLE_HYPERPARAMETER: str = './Resources/Models/GRU/Simple'
PATH_TO_MODEL_GRU_BIDIRECTIONAL_HYPERPARAMETER: str = './Resources/Models/GRU/Bidirectional'

PATH_TO_RESULT_GRU_SIMPLE: str = './Resources/Results/GRU/Simple'
PATH_TO_RESULT_GRU_BIDIRECTIONAL: str = './Resources/Results/GRU/Bidirectional'
PATH_TO_RESULT_TRANSFORMER_BERT: str = './Resources/Results/Transformer/BERT'

MAXIMUM_SENTENCE_LENGTH: int = 344
NUMBER_OF_WORDS = 10000
HYPER_PARAMETER_PROJECT_NAME = 'Hyperparameter Search'

import os
CLEAN_DATA_PATH: str = os.path.normpath(
    os.path.join(
        __file__,
        '../../..',
        'Resources',
        'Data',
        'Clean'
    )
)
