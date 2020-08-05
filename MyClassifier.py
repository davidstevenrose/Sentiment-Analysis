'''
Created on Aug 3, 2020

@author: drose
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import random

class MyClass(object):
    '''
    classdocs
    '''


    def __init__(self, classifier, training_set, testing_set, class_document):
        self.classifier = classifier
        self.training_set = training_set
        self.testing_set = testing_set
        self.class_document = class_document
        self.accuracy = nltk.classify.accuracy(classifier, testing_set)
    
    def getTraining_Set(self):
        return self.training_set
    
    def getTesting_Set(self):
        return self.testing_set
    
    def getClassifier(self):
        return self.classifier
    
    def getAccuracy(self):
        return self.accuracy
    
    def vectorize(self, document):
        words = word_tokenize(document)
        features = {}
        for w in self.class_document:
            features[w] = (w in words)
        return features
        
    def retrain(self, sk = None, force = False):
        fsets = self.training_set + self.testing_set
        random.shuffle(fsets)
        self.training_set = fsets[:len(fsets)-100]
        self.testing_set = fsets[len(fsets)-100:]   
        tempclassifier = None    
        if sk:
            obj = SklearnClassifier(sk)
            tempclassifier = obj.train(self.training_set)
        else:
            tempclassifier = nltk.NaiveBayesClassifier.train(self.training_set)
        tempacc = nltk.classify.accuracy(self.classifier, self.testing_set)
        print(tempacc)
        if tempacc > self.accuracy or force:
            print(str(tempacc)+" > "+str(self.accuracy))
            self.accuracy = tempacc
            self.classifier = tempclassifier        
        