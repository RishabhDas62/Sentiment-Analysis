from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sent1 as s



#consumer key, consumer secret, access token, access secret.
ckey="LG0hjZ5yjauATuiyElJOMNou2"
csecret="ZxAh5ht5axB3T7WptfI7yEYj7Lk7bCMP1L39WKgr3u5XNtwUH3"
atoken="810388947184386048-48I4HKNdTquY4EQZUrKiGvM9gpadh0f"
asecret="ZY1LG75yfxISFmRsTLauEIQWxALtXj5S47nKO6FUgWj8F"

#from twitterapistuff import *

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)

            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)

            if confidence*100 >= 60:
                output = open("twitter-out-1.txt","a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()

            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Donald Trump"])
