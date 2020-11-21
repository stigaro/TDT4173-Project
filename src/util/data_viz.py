#######################################################################################################################

# This script performs the visualization and description for data to observe different features of the dataset.

# The data is read in the form of test train sets and the hyper tuner tunes the model training for different parameters
# defined in 'generation.py' script. The best hyper parameters are saved and used to train the model.

# The results are extracted and saved for the tuned model of simple RNN architecture.

#######################################################################################################################

import plotly.graph_objects as go
import plotly.express as px
import re
import pandas as pd
from wordcloud import WordCloud
from src.util.constants import *
from collections import Counter
import matplotlib.pyplot as plt


# function for reading the raw data
def read_data(): # read train and test data
    train = pd.read_csv(DATA_PATH + '/Raw/Corona_NLP_train.csv', encoding='latin1')
    test = pd.read_csv(DATA_PATH + '/Raw/Corona_NLP_test.csv', encoding='latin1')
    df = train.append(test, sort = False)
    df['Sentiment']= df['Sentiment'].astype(str)
    df['OriginalTweet'] = df['OriginalTweet'].astype(str)
    return df

# function for finding the most commonly used words in the tweets
def common_words(t, max_w): #generate wordmaps of common words in tweets

    # com_words = ''
    # for i in t.text:
    #     i = str(i)
    #     tokens = i.split()
    #
    #     for j in range(len(tokens)): # convert words to lower case
    #         tokens[j] = tokens[j].lower()
    #
    #     com_words += " ".join(tokens) + " "
    #
    # fdist = FreqDist(tokens)
    temp = " ".join(t["OriginalTweet"].tolist())

    wc = WordCloud(width=400, height=400, collocations= False, max_words= max_w).generate(temp)

    plt.figure(figsize=(12, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(DATA_PATH_PREPROCESS + '/' + 'common_words.png')


# function for visualizing different sentiment labels and their occurrence frequency
def sentiment_viz(d): # visualization of the sentiment classes
    df_dist = d['Sentiment'].value_counts()
    fig = go.Figure([go.Bar(x = df_dist.index, y = df_dist.values)]);
    fig.update_layout(title_text = 'Sentiment Distribution');
    fig.show();


# function for finding different hashtags and their occurrence frequency
def hashtags(seq):
    line=re.findall(r'(?<=#)\w+',seq)
    return " ".join(line)

# function for finding different mentions and their occurrence frequency
def get_mentions(s):
    mentions = re.findall("(?<![@\w])@(\w{1,25})", s)
    return " ".join(mentions)

# function for visualizing different mentions and their occurrence frequency
def mentions_viz(df):
    df['mentions'] = df['OriginalTweet'].apply(lambda x: get_mentions(x))
    allMentions = list(df[(df['mentions'] != None) & (df['mentions'] != "")]['mentions'])
    allMentions = [tag.lower() for tag in allMentions]
    mentions_df = dict(Counter(allMentions))
    top_mentions_df = pd.DataFrame(list(mentions_df.items()), columns=['word', 'count']).reset_index(
        drop=True).sort_values('count', ascending=False)[:20]

    fig = px.bar(x=top_mentions_df['word'], y=top_mentions_df['count'],
                 orientation='v',
                 color=top_mentions_df['word'],
                 text=top_mentions_df['count'],
                 color_discrete_sequence=px.colors.qualitative.Bold)

    fig.update_traces(texttemplate='%{text:.2s}', textposition='inside', marker_line_color='rgb(8,48,107)',
                      marker_line_width=2)
    fig.update_layout(width=800, showlegend=False, xaxis_title="Word", yaxis_title="Count")
    fig.show()

# function for visualizing different hashtags and their occurrence frequency
def hashtag_viz(d): # Hashtag visualization
    d['hash'] = d['OriginalTweet'].apply(lambda x:hashtags(x));
    fig = px.bar(d['hash'].value_counts()[1:20], orientation="v", color=d['hash'].value_counts()[1:20],
             color_continuous_scale = px.colors.sequential.Plasma, log_y=True, labels={'value':'Count',
                                                                                       'index':'Hashtags',
                                                                                       'color':'None'});
    fig.update_layout(font_color="black", title_text = "Total Hashtags");

    fig.show()

    # fig.write_image(results_path + '\hashtag.png')

# read data
data = read_data()

# find and save the most common words in the tweets
common_words(data,50)


# data structure
print(data.groupby('Sentiment').describe(include=['O']).T)


# find and save the most common sentiments in the tweets
sentiment_viz(data)

# find and save the most common hashtags in the tweets
hashtag_viz(data)

# find and save the most common mentions in the tweets
mentions_viz(data)

