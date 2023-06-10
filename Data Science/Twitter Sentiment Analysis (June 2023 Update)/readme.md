# Twitter Sentiment Analysis

__Introduction:__

In this project we aim to leverage various technologies and techniques including Twitter API, Python, NLP, SQL, and data visualization to perfrom

__Project Outline:__

- Environment setup and data collection
- GUI creation
- Sentiment Analysis
- Data storage
- Data visualization

__Tools Used:__

- Python
- Tweepy, TextBlob, matplotlib, and Tkinter libraries
- Twitter API
- SQL Database
- Tableau

__Table Content:__

- [Twitter Sentiment Analysis](#twitter-sentiment-analysis)
  - [Environment Setup and Data Collection](#environment-setup-and-data-collection)
  - [Building the GUI](#building-the-gui)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Storing Results in SQL](#storing-results-in-sql)
  - [Visualizing Results in Tableau](#visualizing-results-in-tableau)
  - [conclusion](#conclusion)

## Environment Setup and Data Collection

We need __Tweepy__ for interacting with the Twitter API, __NLTK__ and Transformers for NLP, __Tkinter__ for building the GUI, __sqlalchemy__ for interaction with our SQL database, and __pytablewriter__ for writing to Tableau.

``` console
pip install matplotlib pandas tweepy transformers Tkinter sqlalchemy pytablewriter
```

``` python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sqlalchemy import create_engine
from transformers import pipeline
import pytablewriter
import pandas as pd
import tkinter as tk
import tweepy
```

<span style="color:red">Twitter no longer allows free developer account from accessing their API</span>

We need to create a Twitter Developer Account [here](https://developer.twitter.com/) to obtain the necessary API credentials, APIKey, API Secret Key, Access Token, and Access Token Secret.

Once we have our keys, we can access tweets from the Twitter API.

``` Python
def twitter_connect() -> tweepy.API:
    auth = tweepy.OAuth1UserHandler(
        PARAM_CONSUMER_KEY, PARAM_CONSUMER_SECRET, PARAM_ACCESS_TOKEN, PARAM_ACCESS_TOKEN_SECRET
    )

    api = tweepy.API(auth, wait_on_rate_limit=True)

    return api
```

## Building the GUI

``` python
def build_gui(api: tweepy.API):
    window = tk.Tk()
    window.geometry("800x600")
    window.title("Twitter keyword Analyzer")

    entry = tk.Entry(window, width=30)
    entry.grid(row=0, column=0, padx=10, pady=10)

    label = tk.Label(window, text ="")
    label.grid(row=2, column=0, padx=10, pady=10)

    figure = Figure(figsize=(5, 5), dpi=100)
    plot = figure.add_subplot(1, 1, 1)

    def on_button_press():
        keyword = entry.get()

        tweets = twitter_fetch(api, keyword)

        tweets_sentiment = analyze_tweets(tweets)

        positive_count = sum(1 for sentiment in tweets_sentiment if sentiment['label'] == 'POSITIVE')
        negative_count = sum(1 for sentiment in tweets_sentiment if sentiment['label'] == 'NEGATIVE')

        label['text'] = f'Number of tweets found: {len(tweets)}'

        plot.clear()
        plot.bar(['positive', 'negative'], [positive_count, negative_count])
        
        canvas.draw()
    
    canvas = FigureCanvasTkAgg(figure, window)
    canvas.get_tk_widget().grid(row=2, column=0)

    button = tk.Button(window, text="Search", command=on_button_press)
    button.grid(row=1, column=0, padx=10, pady=10)

    window.mainloop()
```

## Sentiment Analysis

For sentiment analysis, we'll use the pre-trained _BERT_ model from the __Transformers__ library. This model has been fine-tuned on a large corpus of English data and can deliver high-quality sentiment analysis.

``` python
def analyze_tweets(tweets: tweepy.Cursor.items) -> list:
    nlp = pipeline("sentiment-analysis")

    def analyze_sentiment(tweet):
        return nlp(tweet.text)[0]
    
    tweets_sentiment = [analyze_sentiment(tweet) for tweet in tweets]

    return tweets_sentiment
```

## Storing Results in SQL

To store our results, we'll create a SQL database and store each tweet alongside its sentiment

``` python
def save_to_sql(tweets_sentiment):
    engine = create_engine('sqlite://', echo=False)

    df = pd.DataFrame(tweets_sentiment, columns=['tweet', 'sentiment'])

    df.tosql('tweets', con=engine)

    engine.execute("SELECT * FROM tweets").fetchall()

    ## continued in the next section
```

## Visualizing Results in Tableau

We'll write our data to a Tableau data extract file and then load this file in Tableau to perform our visualization

``` python
    ## save_to_sql function

    writer = pytablewriter.TableauDataWriter("tweets.tde")
    
    writer.write_table(df)
```

## conclusion

This project succeeded in creating a Twitter sentiment analysis program using user input.

My plan is to further improve the project later by adding:

- Real-time Analysis
- Geographical Analysis
- Topic Modeling
- Time Series Analysis
- Language Support

In the mean time there a ton more guides, tutorials and information on my:

- [analysistutorial.com](https://analysistutorial.com/)
- [Tutorial Github Repo](https://github.com/Arash-Nozarinejad/analysis-tutorial)
- [Projects Github Repo](https://github.com/Arash-Nozarinejad/public-projects)
- [Twitter](https://twitter.com/analysistut)
- [Youtube](https://www.youtube.com/@analysistutorial)

You can also connect with me on Linkedin:

- [LinkedIn](https://www.linkedin.com/in/arash-nozarinejad/)
