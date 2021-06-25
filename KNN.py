import pyprind
import pandas as pd
import os
import re
import numpy as np
import gensim
import gensim.models
from sklearn.neighbors import KNeighborsClassifier
import time

class KNN():
    def __init__(self, vecs, labels):
        self.vecs = vecs
        self.labels = labels

    def KNN_pred(self):
        clf = KNeighborsClassifier(n_neighbors=2,weights="distance",p=1)  # class 
        clf.fit(self.vecs[0:25000],self.labels[0:25000])
        pbar=pyprind.ProgBar(50000)
        correct=0
        for i in range(25000,50000):
            test=np.array(self.vecs[i]).reshape([1,-1])
            result = clf.predict(test)
            if result==self.labels[i]:
                correct+=1
            pbar.update()
        print('\naccuracy=',correct/25000)
        print("Total time:",time.time()-start_time)
        return result, correct/25000

