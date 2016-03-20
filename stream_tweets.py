# -*- coding: utf-8 -*-
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import StreamListener
import json
import testing as s

consumerkey = 'KuMYXgdFWRv5cbGx4iVUhrsdb'
secretkey =  '1cI2FPyzM8ru2yvWmhuw7a9ZZQiKqkI98yT0OrG0q4CRmkyNH5'
access_token = '162372886-42v1bc1RWkpnR3WXMIs8bMOiVAycrdf1EiWKWsth' 
access_secret = 'm24l1Z1LCyVA93VOTPte1XFvFvfBFGFBDkgNTOEw5KEb5'

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
#        user = all_data['user']
#        print (user['location'])
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
#        print (all_data['text'])
        
    def on_error(self, status):
        print(status)

auth = OAuthHandler(consumerkey, secretkey)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Trump"])