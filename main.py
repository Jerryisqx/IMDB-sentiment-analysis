import numpy as np
from SVM import SVM
from SVM import SVM1
from tf_idf import tf_idf
from NaiveBayes import NaiveBayes
from logistics import logistics
#from glove import Glove
from KNN import KNN
from word2vec import Word2Vec
from hash_logistic import hash_logistic
import time


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

def tfidf_SVM(): 
    tfidf = tf_idf()
    tfidf.get_data()
    train_X, train_Y, test_X, test_Y = tfidf.train_X, tfidf.train_Y, tfidf.test_X, tfidf.test_Y
    svm = SVM1(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    svm_pred, svm_acc = svm.TFIDF_SVM()

def tfidf_KNN():
    tfidf = tf_idf()
    tfidf.get_data()
    train_X, train_Y, test_X, test_Y = tfidf.train_X, tfidf.train_Y, tfidf.test_X, tfidf.test_Y
    Knn = KNN(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    Knn_pred, Knn_acc = Knn.tfidf_SVM_pred()

def tfidf_logistic():
    tfidf = tf_idf()
    tfidf.get_data()
    train_X, train_Y, test_X, test_Y = tfidf.train_X, tfidf.train_Y, tfidf.test_X, tfidf.test_Y
    logis = logistics(np.r_[train_X, test_X], np.r_[train_Y, test_Y])
    prediction, acc = logis.logistics_pred()


'''
def Glove_logistics():
    g=Glove()
    g.open_file()
    g.get_data()
    g.formlabels()
    g.form_wordmatrix()
    g.form_Glove()
    g.form_mittens()
    logis = logistics(g.vecs, g.labels)
    prediction, acc = logis.logistics_pred()

def Glove_KNN():
    g=Glove()
    g.open_file()
    g.get_data()
    g.formlabels()
    g.form_wordmatrix()
    g.form_Glove()
    g.form_mittens()
    model2=g.get_Glove_model()
    word2vec=Word2Vec()
    data,labels,model=word2vec.form_word2vec()
    Knn=KNN(labels=labels,model1=model,model2=model2)
    Knn.Glove_KNN()
'''

def w2v_SVM():
    word2vec=Word2Vec()
    data,labels,model=word2vec.form_word2vec()
    Svm=SVM(data=data,labels=labels,model1=model)
    Svm.W2V_SVM()

def w2v_KNN():
    word2vec=Word2Vec()
    data,labels,model=word2vec.form_word2vec()
    Knn=KNN(labels=labels,model1=model)
    Knn.W2V_KNN()

def hashLogistic(): 
    HashLogistic = hash_logistic()
    HashLogistic.read_data()
    HashLogistic.predict()

def glove_SVM():
    g=Glove()
    g.open_file()
    g.get_data()
    g.formlabels()
    g.form_wordmatrix()
    g.form_Glove()
    g.form_mittens()
    model2=g.get_Glove_model()



#tfidf_NB()      #GaussianNB 0.74312; MultinomialNB 0.829; BernoulliNB 0.82316
# tfidf_SVM()
#tfidf_logistic() #accuracy= 0.8699043885266232
t1 = time.time()
w2v_SVM()       #accuracy= 0.70252
t2 = time.time()
print(t2-t1)
#hashLogistic()  #accuracy: 0.881


