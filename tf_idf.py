import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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
