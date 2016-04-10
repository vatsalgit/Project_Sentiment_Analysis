# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:41:25 2016

@author: Vatsal Shah
"""

import nltk
from nltk.tokenize import word_tokenize

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()
    

all_words = []
reviews = []
    


for p in short_pos.split('\n'):
    words = word_tokenize(p)
    for w in words:
        all_words.append(w.lower())

    
for p in short_neg.split('\n'):
    words = word_tokenize(p)
    for w in words:
        all_words.append(w.lower())

freq = nltk.FreqDist(all_words)

import csv

with open('word_frequency.csv', 'w') as csvfile:
    
    w = csv.writer(csvfile)
    w.writerows(freq.items())
        
    


    