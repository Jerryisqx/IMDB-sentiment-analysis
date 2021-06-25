import pyprind
import pandas as pd
import os
import re
import numpy as np
import gensim
import gensim.models
from sklearn.linear_model import LogisticRegression
import time

class logistics():
    def __init__(self, vecs, labels):
        self.vecs = vecs
        self.labels = labels
    def logistics_pred(self):
        clf = LogisticRegression(tol=1e-4,C=3) 
        clf.fit(self.vecs[24997:49987],np.array(self.labels[24997:49987]))
        pbar=pyprind.ProgBar(50000)
        correct=0
        for i in range(0,24997):
            test=np.array(self.vecs[i]).reshape([1,-1])
            result = clf.predict(test)
            if result==self.labels[i]:
                correct+=1
            pbar.update()
        print('\naccuracy=',correct/24997)
        return result, correct/24997

