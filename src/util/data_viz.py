import plotly.graph_objects as go
import plotly.express as px
import re
import pandas as pd
from wordcloud import WordCloud
from src.util.constants import *
from collections import Counter
import matplotlib.pyplot as plt



def read_data(): # read train and test data
    train = pd.read_csv(DATA_PATH + '/Raw/Corona_NLP_train.csv', encoding='latin1')
    test = pd.read_csv(DATA_PATH + '/Raw/Corona_NLP_test.csv', encoding='latin1')
    df = train.append(test, sort = False)
    df['Sentiment']= df['Sentiment'].astype(str)
    df['OriginalTweet'] = df['OriginalTweet'].astype(str)
    return df

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


def sentiment_viz(d): # visualization of the sentiment classes
    df_dist = d['Sentiment'].value_counts()
    fig = go.Figure([go.Bar(x = df_dist.index, y = df_dist.values)]);
    fig.update_layout(title_text = 'Sentiment Distribution');
    fig.show();


def hashtags(seq):
    line=re.findall(r'(?<=#)\w+',seq)
    return " ".join(line)

def get_mentions(s):
    mentions = re.findall("(?<![@\w])@(\w{1,25})", s)
    return " ".join(mentions)

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

# data = data[data["Sentiment"]=="Positive"]['OriginalTweet']
common_words(data,50)

#
#
# data structure
print(data.groupby('Sentiment').describe(include=['O']).T)


# sentiment visualization
sentiment_viz(data)

# hashtag visualization
hashtag_viz(data)

#mentions visualization
mentions_viz(data)

