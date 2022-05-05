from asyncio.windows_events import NULL
import tweepy
import os
import openai
from dotenv import load_dotenv

load_dotenv()

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

openai.api_key = os.getenv("OPENAI_API_KEY")

client = tweepy.Client(TWITTER_BEARER_TOKEN, TWITTER_API_KEY,
                       TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN,
                       TWITTER_ACCESS_TOKEN_SECRET)

query = '"ETH" OR "Ethereum"'
tweets = client.search_recent_tweets(query=query, max_results=100)
sentiments = []

for tweet in tweets.data:
    response = openai.Completion.create(engine="text-davinci-002",
                                        prompt="Classify the sentiment in these tweets:\n" +
                                        tweet.text + "\nSentiment:",
                                        temperature=0,
                                        max_tokens=134,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0)
    #print(response.choices[0]['text'])
    sentiments.append(response.choices[0]['text'].strip())

#print(sentiments)
sentiment_score = 0
for sentiment in sentiments:
    if sentiment == 'Positive':
        sentiment_score += 1
    elif sentiment == 'Happy':
        sentiment_score += 1
    elif sentiment == 'Negative':
        sentiment_score -= 1
    elif sentiment == 'Sad':
        sentiment_score -= 1
    elif sentiment == 'Bullish':
        sentiment_score -= 1
    elif sentiment == 'Neutral':
        sentiment_score += 0
    
print(f'The sentiment score for {query} is: ' + str(sentiment_score))