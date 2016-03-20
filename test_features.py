# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:57:08 2016

@author: Vatsal Shah
"""

"""
Work to be done:
1. Add more algorithms
2. Seperate the sentiment function into another file
3. Use 10,000 best word features instead of 15000
4. Develop a simple GUI if possible
5. Connect with Twitter API #IMPORTANT
6. Generate data in csv format to plot graphs and dashboard for RShiny
7. Negation Handling if possible 

"""

import random
# -*- coding: utf-8 -*-
#File: sentiment_mod.py
#from nltk.corpus import movie_reviews
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
#from final_demo_v2 import best_words


"""
Pickle the best words
save_words = open("pickled_algos/best_words15k.pickle","wb")
pickle.dump(best_words, save_words)
save_words.close()
"""
# Load the best words
best_word_features = open("pickled_algos/best_words15k.pickle", "rb")
best_words_15k = pickle.load(best_word_features)
best_word_features.close()


test_string = "The gift I received on my birthday was wonderful"

def find_features(text):
    words = word_tokenize(text)
    features = {}
    for w in words:
        features[w] = (w in best_words_15k)

    return features

def sentiment (text):
    feats = find_features(text)
    print(voted_classifier.classify(feats),voted_classifier.confidence(feats))


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


open_file = open("pickled_algos/originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)



sentiment(test_string)

