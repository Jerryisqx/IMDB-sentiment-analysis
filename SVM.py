import os
from sklearn import svm
import numpy as np

class SVM():
    def __init__(self,data=None,labels=None,model1=None,model2=None,train_X=None, train_Y=None, test_X=None, test_Y=None,vec=np.zeros([50000,100]),vec2=np.zeros([50000,100])):
        self.data=data
        self.labels=labels
        self.w2v_model=model1
        self.glove_model=model2
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.vec=vec
        self.vec2=vec2

    def TFIDF_SVM(self):
        svmmodel = svm.SVC(C = 15.0 , kernel= "linear") 
        nn = svmmodel.fit(self.train_X, self.train_Y)
        pre_test = svmmodel.predict(self.test_X)
        correct=0
        for i in range(25000):
            if pre_test[i] == self.test_Y[i]:
                correct += 1
        acc = correct/25000
        print('the acc of tfidf+svm is', acc)
        return pre_test, acc
        
    def get_w2v(self):
        self.labels=np.array(self.labels).reshape([50000,1])
        for i in range(50000):
            count=0
            sent=self.data[i]
            for j in range(len(sent)):
                #print(len(sent))
                try:
                    count += 1
                    self.vec[i,0:100]+=self.w2v_model.wv[sent[j]]
                    #print(self.w2v_model.wv[sent[j]])
                except KeyError:
                    continue
            self.vec[i,0:100] /= count
        print(self.vec)

    def W2V_SVM(self):
        self.get_w2v()
        clf = svm.SVC()  
        clf.fit(self.vec[0:25000,0:100], self.labels[0:25000,0])
        correct=0
        result = clf.predict(self.vec[25000:50000,0:100])
        #print(type(result),result.shape)
        for i in range(12500):
            if result[i,]==1:
                correct+=1
        for i in range(12500,25000):
            if result[i,]==0:
                correct+=1
        print('accuracy=',correct/25000)

class SVM1():
    def __init__(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
    def TFIDF_SVM(self):
        svmmodel = svm.SVC(C = 15.0 , kernel= "linear") #kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";
        nn = svmmodel.fit(self.train_X, self.train_Y)
        pre_test = svmmodel.predict(self.test_X)
        correct=0
        for i in range(25000):
            if pre_test[i] == self.test_Y[i]:
                correct += 1
        acc = correct/25000
        print('the acc of tfidf+svm is', acc)
        return pre_test, acc

