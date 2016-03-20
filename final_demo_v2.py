# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:40:40 2016

@author: Vatsal Shah
"""
from evaluate_features import evaluate_features,make_full_dict
from train_classifiers import train_classifiers
from word_scoring import create_word_scores,find_best_words

def best_word_features(words):          
	return dict([(word, True) for word in words if word in best_words])   
 
#tries using all words as the feature selection mechanism
print ('using all words as features')
posFeatures,negFeatures=evaluate_features(make_full_dict)
trainFeatures,testFeatures = train_classifiers(posFeatures,negFeatures)

#finds word scores
word_scores = create_word_scores()

#tries the best_word_features mechanism with each of the numbers_to_test of features
numbers_to_test = [10, 100, 1000, 10000, 15000]


#creates feature selection mechanism that only uses best words


for num in numbers_to_test:
    print ('evaluating best %d word features' % (num))
    best_words = find_best_words(word_scores, num)
    posFeatures,negFeatures = evaluate_features(best_word_features)
    trainFeatures,testFeatures = train_classifiers(posFeatures,negFeatures)

