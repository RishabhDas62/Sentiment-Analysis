from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time

ckey="LG0hjZ5yjauATuiyElJOMNou2"
csecret="ZxAh5ht5axB3T7WptfI7yEYj7Lk7bCMP1L39WKgr3u5XNtwUH3"
atoken="810388947184386048-48I4HKNdTquY4EQZUrKiGvM9gpadh0f"
asecret="ZY1LG75yfxISFmRsTLauEIQWxALtXj5S47nKO6FUgWj8F"

class listener(StreamListener):

    def on_data(self,data):
        try:
            #print(data)
            tweet=data.split(',"text":"')[1].split('","source')[0]
            print(tweet)
            saveThis=str(time.time())+"::"+tweet
            
            
            saveFile=open("twitDB.csv","a")
            saveFile.write(saveThis)
            saveFile.write("\n")
            saveFile.close()
            return True
        except (BaseException,e):
            print("failed ondata",str(e))
            time.sleep(5)

    def on_error(self,status):
        print("on error",status)



auth=OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)
twitterStream=Stream(auth,listener())

twitterStream.filter(track=["Donald Trump"])
