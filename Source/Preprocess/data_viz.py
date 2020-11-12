import plotly.graph_objects as go
import plotly.express as px
import re
import Source.Preprocess.data_utility as du
from sklearn import metrics

results_path = r".\Resources\Results"


def sentiment_viz(d): # visualization of the sentiment classes
    df_dist = d['Sentiment'].value_counts()
    fig = go.Figure([go.Bar(x = df_dist.index, y = df_dist.values)]);
    fig.update_layout(title_text = 'Sentiment Distribution');
    fig.show();


def hashtags(seq):
    line=re.findall(r'(?<=#)\w+',seq)
    return " ".join(line)


def hashtag_viz(d): # Hashtag visualization
    d['hash'] = d['OriginalTweet'].apply(lambda x:hashtags(x));
    fig = px.bar(d['hash'].value_counts()[1:20], orientation="v", color=d['hash'].value_counts()[1:20],
             color_continuous_scale = px.colors.sequential.Plasma, log_y=True, labels={'value':'Count',
                                                                                       'index':'Hashtags',
                                                                                       'color':'None'});
    fig.update_layout(font_color="black", title_text = "Total Hashtags");

    fig.show()

    # fig.write_image(results_path + '\hashtag.png')


# def data_metrics:
    # ROC curve
    # to be added soon

# read data
data = du.read_data();

# data structure
print(data.groupby('Sentiment').describe(include=['O']).T)


# sentiment visualization
sentiment_viz(data)

# hashtag visualization
hashtag_viz(data)


