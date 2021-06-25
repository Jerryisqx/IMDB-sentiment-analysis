import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes():
    def __init__(self,train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
    # predict with NB
    def pred_gaussianNB(self): 
        gnb = GaussianNB().fit(self.train_X, self.train_Y)
        gnb_prediction = gnb.predict(self.test_X)
        gnb_acc_score = gnb.score(self.test_X, self.test_Y)
        return gnb_prediction, gnb_acc_score
    def pred_multinomialNB(self):
        mnb = MultinomialNB(alpha=0.001).fit(self.train_X, self.train_Y)
        mnb_prediciton = mnb.predict(self.test_X)
        mnb_acc_score = mnb.score(self.test_X, self.test_Y)
        return mnb_prediciton, mnb_acc_score
    def pred_bernoulliNB(self):
        bnb = BernoulliNB(alpha=0.001).fit(self.train_X, self.train_Y)
        bnb_prediction = bnb.predict(self.test_X)
        bnb_acc_score = bnb.score(self.test_X, self.test_Y)
        return bnb_prediction, bnb_acc_score
