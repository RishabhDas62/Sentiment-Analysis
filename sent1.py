import nltk
import random
from nltk.tokenize import word_tokenize

#from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
#from sklearn.linear_model import LogisticRegression,SGDClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)

        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf
        


documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

#short_pos=open("short_reviews/positive.txt",encoding = "ISO-8859-1").read()
#short_neg=open("short_reviews/negative.txt",encoding = "ISO-8859-1").read()

#all_words=[]
#documents=[]

#allowed_word_types=["J"]

"""for p in short_pos.split("\n"):
    documents.append((p,"pos"))
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split("\n"):
    documents.append((p,"neg"))
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())"""


#save_documents = open("documents.pickle","wb")
#pickle.dump(documents, save_documents)
#save_documents.close()



#documents=[(list(movie_reviews.words(fileid)),category)
#           for category in movie_reviews.categories()
#           for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)

#print(documents[-1])

#all_words=[]

#short_pos_words=word_tokenize(short_pos)
#short_neg_words=word_tokenize(short_neg)

#for w in short_pos_words:
#    all_words.append(w.lower())
#for w in short_neg_words:
#    all_words.append(w.lower())
    

##for w in movie_reviews.words():
##    all_words.append(w.lower())


#all_words=nltk.FreqDist(all_words)

#word_features=list(all_words.keys())[:5000]

#save_word_features = open("word_features5k.pickle","wb")
#pickle.dump(word_features, save_word_features)
#save_word_features.close()

#print(word_features)
word_features5k_f = open("word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features



featuresets_f = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()
random.shuffle(featuresets)
print(len(featuresets))


#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
#featuresets=[(find_features(rev),category) for (rev,category) in documents]
#save_featuresets = open("featuresets.pickle","wb")
#pickle.dump(featuresets,save_featuresets)
#save_featuresets.close()
#print("done")
#positive data
#training_set=featuresets[:1900]
#testing_set=featuresets[1900:]
#neagtive data
training_set=featuresets[:10000]
testing_set=featuresets[10000:]


#classifier=nltk.NaiveBayesClassifier.train(training_set)
classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

#print("Naive Bayes accuracy %:", (nltk.classify.accuracy(classifier,training_set))*100)
#classifier.show_most_informative_features(15)

#save_classifier=open("naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()

#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)
MNB_classifier_f=open("MNB_classifier.pickle","rb")
MNB_classifier=pickle.load(MNB_classifier_f)
MNB_classifier_f.close()


#print("MultinomialNB accuracy %:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)

#save_classifier=open("MNB_classifier.pickle","wb")
#pickle.dump(MNB_classifier,save_classifier)
#save_classifier.close()



#BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier_f=open("BNB_classifier.pickle","rb")
BNB_classifier=pickle.load(BNB_classifier_f)
BNB_classifier_f.close()


#BNB_classifier.train(training_set)
#print("BernoulliNB accuracy %:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)
#save_classifier=open("BNB_classifier.pickle","wb")
#pickle.dump(BNB_classifier,save_classifier)
#save_classifier.close()



voted_classifier=VoteClassifier(classifier,MNB_classifier,BNB_classifier)

#print("Voted Classifier Accuracy %",(nltk.classify.accuracy(voted_classifier,testing_set))*100)

#print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

def sentiment(text):
    feats=find_features(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
