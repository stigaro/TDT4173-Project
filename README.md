# Sentiment Analysis on Covid19 twitter dataset
Repository for the TDT4173-Project (Machine Learning)
----------------------------------------------------------- 
1- Structure

The project has been divided into Resources and src folders.

- Resources contain the Data, Trained Models, Notebooks and Results.

- src has the scripts for running the model training and evaluation.
It also contains the utility functions for data loading, pre-processing
visualization and helper functions.

2- Experiments

The following scripts can be used to train models for different methods:

- 'lstm' and 'lstm_hypersearch' for LSTM

- 'run_rnn_simple' and 'run_rnn_bidirect' for RNN

- 'run_gru_simple' and 'run_gru_bidirectional' for GRU

- 'run_mlp' for MLP

- 'run_bert' for BERT

3- Utilities

The following scripts are utilized for defining th require utility functions:

- 'constants' for declaring commonly used constant variables like root directory

- 'data' for pre processing and reading the dataset

- 'data_viz' for pre processing and data description visualizations

- 'extraction' for performance evaluation metrics extraction from the trained models

- 'generation' for building models for different sequential and baseline methods

- 'loading' for data extraction and pre processing from the dataset

- 'normalizers' for data cleaning and normalization operations on loaded data

- 'tokenizers' for tokenization operations during the data cleaning

- 'util' for different miscellaneous utility functions

- 'vectorizers' for different vectorization operations on data










