# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:40:40 2016

@author: Vatsal Shah
"""
import math, collections
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
from statistics import mode
import nltk
import pickle

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
     

def train_classifiers(posFeatures,negFeatures):
    
    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    
    #trains a Naive Bayes Classifier
    print ("----------------Naive Bayes Classifier-----------")
    classifier = NaiveBayesClassifier.train(trainFeatures)	
    
    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)	
    
    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
    	referenceSets[label].add(i)
    	predicted = classifier.classify(features)
    	testSets[predicted].add(i)	
    
    #prints metrics to show how well the feature selection did
    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print ('Original Naive Bayes Accuracy:', (nltk.classify.util.accuracy(classifier, testFeatures))*100)
    print ('pos precision:', precision(referenceSets['pos'], testSets['pos']))
    print ('pos recall:', recall(referenceSets['pos'], testSets['pos']))
    print ('neg precision:',precision(referenceSets['neg'], testSets['neg']))
    print ('neg recall:', recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)

    #Pickle the algorithm for future use
    save_classifier = open("pickled_algos/originalnaivebayes.pickle","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()    
     
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(trainFeatures)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testFeatures))*100)

    #Pickle the algorithm for future use    
    save_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
    pickle.dump(MNB_classifier, save_classifier)
    save_classifier.close()   

    
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(trainFeatures)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testFeatures))*100)
    
    #Pickle the algorithm for future use     
    save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
    pickle.dump(BernoulliNB_classifier, save_classifier)
    save_classifier.close()    
    
    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(trainFeatures)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testFeatures))*100)
    
    #Pickle the algorithm for future use 
    save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(trainFeatures)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testFeatures))*100)
    
    #Pickle the algorithm for future use    
    save_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()

    
    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(trainFeatures)
    print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testFeatures)*100)
    
    #Pickle the algorithm for future use 
    save_classifier = open("pickled_algos/SGDC_classifier.pickle","wb")
    pickle.dump(SGDC_classifier, save_classifier)
    save_classifier.close()
    
    Dec_Tree_Classifier = SklearnClassifier(DecisionTreeClassifier())    
    Dec_Tree_Classifier.train(trainFeatures)
    print("DecisionTreeClassifier Accuracy:",(nltk.classify.accuracy(Dec_Tree_Classifier,testFeatures))*100)
    
    
    #Pickle the algorithm for future use 
    save_classifier = open("pickled_algos/decision_tree.pickle","wb")
    pickle.dump(Dec_Tree_Classifier, save_classifier)
    save_classifier.close()    
    
    """
    
#    Grad_Boost_Classifier = SklearnClassifier(GradientBoostingClassifier())
#    Grad_Boost_Classifier.train(trainFeatures)
#    print("Gradient Boosting Classifier Accuracy:", (nltk.classify.accuracy(Grad_Boost_Classifier,testFeatures))*100)    
    """    
    
    Random_Forest_Classifier = SklearnClassifier(RandomForestClassifier())
    Random_Forest_Classifier.train(trainFeatures)
    print("Random Forest Classifier Accuracy:",(nltk.classify.accuracy(Random_Forest_Classifier,testFeatures
    ))*100)
    
    #Pickle the algorithm for future use 
    save_classifier = open("pickled_algos/random_forest.pickle","wb")
    pickle.dump(Random_Forest_Classifier, save_classifier)
    save_classifier.close()
    
    Ada_Boost_Classifier = SklearnClassifier(AdaBoostClassifier())
    Ada_Boost_Classifier.train(trainFeatures)
    print("Ada Boost Classifier Accuracy:",(nltk.classify.accuracy(Ada_Boost_Classifier,testFeatures))*100) 
    
    #Pickle the algorithm for future use 
    save_classifier = open("pickled_algos/Ada_Boost.pickle","wb")
    pickle.dump(Ada_Boost_Classifier, save_classifier)
    save_classifier.close()
    
    
    voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  Random_Forest_Classifier,
                                  Ada_Boost_Classifier
                                  )
                                                    
    print("Voted classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testFeatures))*100) 
    
    # The voted classifier could not be pickled. Check this later!    
    
    
    return trainFeatures,testFeatures
