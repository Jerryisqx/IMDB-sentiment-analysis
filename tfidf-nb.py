import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import time

class tf_idf():
    def __init__(self):
        self.train_X = None 
        self.train_Y = None 
        self.test_X = None 
        self.test_Y = None

    def read_file(self, path):
        files= os.listdir(path) 
        txt = []
        for file in files: 
            position = path+'/'+ file 
            #print (position)           
            with open(position, "r",encoding='utf-8') as f:    
                data = f.read()
                txt.append(data)
        return txt

    def get_data(self):
        pos_train = self.read_file('data/train/pos')
        neg_train = self.read_file('data/train/neg')
        pos_test = self.read_file('data/test/pos')
        neg_test = self.read_file('data/test/neg')

        # transfer the data into tfidf vectors. 
        vectorizer = CountVectorizer(max_features=3000)
        tf_idf_transformer = TfidfTransformer()
        tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(pos_train))
        pos_train_vec = tf_idf.toarray()
        tf_idf = tf_idf_transformer.transform(vectorizer.transform(neg_train))
        neg_train_vec = tf_idf.toarray()  
        tf_idf = tf_idf_transformer.transform(vectorizer.transform(pos_test))
        pos_test_vec = tf_idf.toarray()  
        tf_idf = tf_idf_transformer.transform(vectorizer.transform(neg_test))
        neg_test_vec = tf_idf.toarray()  

        # get test_X & test_Y & train_X & train_Y
        self.train_X = np.r_[pos_train_vec, neg_train_vec]
        self.test_X = np.r_[pos_test_vec, neg_test_vec]
        self.train_Y = np.array([1]*pos_train_vec.shape[0] + [0]*neg_train_vec.shape[0])
        self.test_Y = np.array([1]*pos_test_vec.shape[0] + [0]*neg_test_vec.shape[0])

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


def tfidf_NB(): 
    tfidf = tf_idf()
    tfidf.get_data()
    train_X, train_Y, test_X, test_Y = tfidf.train_X, tfidf.train_Y, tfidf.test_X, tfidf.test_Y
    NB = NaiveBayes(train_X, train_Y, test_X, test_Y)
    gnb_prediction, gnb_acc_score = NB.pred_gaussianNB()
    mnb_prediciton, mnb_acc_score = NB.pred_multinomialNB()
    bnb_prediction, bnb_acc_score = NB.pred_bernoulliNB()
    print('The accuracy of GaussianNB is', gnb_acc_score)
    print('The accuracy of MultinomialNB is', mnb_acc_score)
    print('The accuracy of BernoulliNB is', bnb_acc_score)

t1 = time.time()
tfidf_NB()
t2 = time.time()
print(t2-t1)
