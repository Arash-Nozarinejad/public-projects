from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sqlalchemy import create_engine
from transformers import pipeline
import pytablewriter
import pandas as pd
import tkinter as tk
import tweepy

#Sno4UTU3VXhJMXlNWnZZMnpKUFk6MTpjaQ
#bQ5Rt7Kl9k9uUzITGlRwDEBzHy0vvJb7G1MmfeHf7rj84rNoFl

#PARAM_CONSUMER_KEY = '7n6HXTXcaObMMos8Vbyet8YG1'
#PARAM_CONSUMER_SECRET = 'kUDhLI7Yf5ngIjg8yrt5ZmeEV8SmPvLtzcjP6ZgP55RfmXXWer'
PARAM_CONSUMER_KEY = '7n6HXTXcaObMMos8Vbyet8YG1'
PARAM_CONSUMER_SECRET = 'kUDhLI7Yf5ngIjg8yrt5ZmeEV8SmPvLtzcjP6ZgP55RfmXXWer'
PARAM_ACCESS_TOKEN = '1613468975580565505-ek1RLUZWKqRs7egTat4ku1m3muldX4'
PARAM_ACCESS_TOKEN_SECRET = 'IIuuXCQcHSJ7Cz47M3iCoWWeqryVtpFMP6kmoZyz0QdC1'

def twitter_connect() -> tweepy.API:
    auth = tweepy.OAuth1UserHandler(
        PARAM_CONSUMER_KEY, PARAM_CONSUMER_SECRET, PARAM_ACCESS_TOKEN, PARAM_ACCESS_TOKEN_SECRET
    )

    api = tweepy.API(auth, wait_on_rate_limit=True)

    return api



def twitter_fetch(api: tweepy.API, keyword: str):
    tweets = api.search_tweets(q=search_query, lang="en", count=100, tweet_mode ='extended')

    return tweets


def analyze_tweets(tweets: tweepy.Cursor.items) -> list:
    nlp = pipeline("sentiment-analysis")

    def analyze_sentiment(tweet):
        return nlp(tweet.text)[0]
    
    tweets_sentiment = [analyze_sentiment(tweet) for tweet in tweets]

    return tweets_sentiment

def save_to_sql(tweets_sentiment):
    engine = create_engine('sqlite://', echo=False)

    df = pd.DataFrame(tweets_sentiment, columns=['tweet', 'sentiment'])

    df.tosql('tweets', con=engine)

    engine.execute("SELECT * FROM tweets").fetchall()

    writer = pytablewriter.TableauDataWriter("tweets.tde")
    
    writer.write_table(df)

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

        save_to_sql(tweets_sentiment)
    
    canvas = FigureCanvasTkAgg(figure, window)
    canvas.get_tk_widget().grid(row=2, column=0)

    button = tk.Button(window, text="Search", command=on_button_press)
    button.grid(row=1, column=0, padx=10, pady=10)

    window.mainloop()


def run():
    api = twitter_connect()
    build_gui(api)


if __name__ == '__main__':
    # run()
    search_query = "'ref''world cup'-filter:retweets AND -filter:replies AND -filter:links"
    api = twitter_connect()
    tweets = twitter_fetch(api, search_query)
    for tweet in tweets:
        print(tweet.text)
