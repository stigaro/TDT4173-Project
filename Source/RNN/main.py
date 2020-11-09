import Source.Preprocess.data_utility as du

import tensorflow.keras.models as kr
from sklearn.model_selection import train_test_split

from Source.RNN import RNN_train, RNN_eval


# read raw input files
data = du.read_data();

print(data.shape)


# preprocess data
tweet = data['OriginalTweet'].copy()
senti = data['Sentiment'].copy()


tweet_cleaned = tweet.apply(du.clean_data) # clean
[tweet_token, vocab_size] = du.token(tweet_cleaned) # tokenize
tweet_pad = du.padd(tweet_token)

print(vocab_size)

# label encoding for reducing ambiguity
senti, enco = du.enco(data)




# test and train split from data
t_train, t_test, s_train, s_test = train_test_split(tweet_pad, senti, test_size=0.3, stratify=senti)


#print(tweet_pad.shape)
print(t_train.shape, s_train.shape)
print(t_test.shape, s_test.shape)


###########################################Training or Model Loading

val = input("Enter 0 to train and 1 to load model")

model_path = r".\Resources\Models\RNN"

if val!='1' :
    # simple RNN model training
    rnn_s = RNN_train.simple_rnn(tweet_pad, t_train, s_train, vocab_size)
    # bidirectional RNN model training
    rnn_b = RNN_train.bidirect_rnn(tweet_pad, t_train, s_train, vocab_size)
else:
    rnn_s = kr.load_model(model_path + r"\simpleRNN_bcrossentropy")
    rnn_b = kr.load_model(model_path + r"\biRNN_bcrossentropy")

##########################################Evaluation
# simple RNN model evaluation
RNN_eval.confusion_rnn(rnn_s, t_test, s_test, enco.classes_)
# bidirectional RNN model evaluation
RNN_eval.confusion_rnn(rnn_b, t_test, s_test, enco.classes_)









