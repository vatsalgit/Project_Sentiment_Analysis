# -*- coding: utf-8 -*-
import nltk
import random
#from nltk.corpus import movie_reviews

import pickle
from nltk.tokenize import word_tokenize

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()
    

all_words = []
reviews = []
    
#j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    reviews.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    reviews.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_reviews = open("pickled_algos/reviews.pickle","wb")
pickle.dump(reviews, save_reviews)
save_reviews.close()


freq = nltk.FreqDist(all_words)

freq = freq.most_common()

word_features = [freq[key][0] for key in range(0,4000)]


save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(review):
    words = word_tokenize(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

    
featuresets = [(find_features(rev), category) for (rev, category) in reviews]

random.shuffle(featuresets)

save_featuresets = open("pickled_algos/final_featuresets.pickle","wb")
pickle.dump(featuresets,save_featuresets)
save_featuresets.close()

