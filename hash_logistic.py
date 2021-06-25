import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import pyprind
pbar = pyprind.ProgBar(50000)
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import HashingVectorizer
import pyprind
import numpy as np
import warnings


class hash_logistic():
    def __init__(self):
        stop = None
    def read_data(self):
        # 1. read the data from the files and restore to a csv file.
        labels = {'pos':1, 'neg':0}
        df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = 'data/{}/{}'.format(s, l)
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r',encoding='UTF-8') as infile:
                        txt = infile.read()
                    df = df.append( [ [txt, labels[l]] ], ignore_index=True )
                    pbar.update()
        df.columns = ['review', 'sentiment']
        from sklearn.utils import shuffle
        df[0:25000] = shuffle(df[0:25000])
        df[25000:]=shuffle(df[25000:])
        df.to_csv('./movie_data.csv', index = False)
        df = pd.read_csv('./movie_data.csv')
        print('df_to_csv done!')
 
        # preprocessing the text
        self.stop = stopwords.words('english')

    def tokenizer(self, text):
        text=re.sub('<[^>]*>','',text)
        emoticons=re.findall('(?::|;|=)(?:-)?(?:</span>|<spanclass="es0">|D|P)',text.lower())
        text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')
        tokenized=[w for w in text.split() if w not in stop]
        return tokenized

    def stream_docs(self, path):
        with open(path, 'r', encoding='UTF-8') as csv:
            next(csv)                  #skip header
            for line in csv:
                text, label = line[:-3], int(line[-2])
                yield text, label

    def get_minibatch(self, doc_stream, size):
        docs, y = [], []
        try:
            for _ in range(size):
                text, label = next(doc_stream)
                docs.append(text)
                y.append(label)
        except StopIteration:
            return None,None
        return docs, y

    def predict(self):
        def tokenizer(text):
            text=re.sub('<[^>]*>','',text)
            emoticons=re.findall('(?::|;|=)(?:-)?(?:</span>|<spanclass="es0">|D|P)',text.lower())
            text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')
            tokenized=[w for w in text.split() if w not in self.stop]
            return tokenized

        vect = HashingVectorizer( decode_error='ignore', n_features=3000000, preprocessor=None, tokenizer=tokenizer)
        clf = LogisticRegression(tol=1e-4,C=3)    # , n_iter=1
        warnings.filterwarnings('ignore')

        doc_stream = self.stream_docs(path='./movie_data.csv')
        x,y = self.get_minibatch(doc_stream, size=50000)
        x_test=x[:25000]
        x_train=x[25000:]
        y_test=y[:25000]
        y_train=y[25000:]
        x_train = vect.transform(x_train)
        x_test = vect.transform(x_test)
        clf.fit(x_train,y_train)
        pred = clf.predict(x_test)
        print('accuracy: %.3f'%clf.score(x_test,y_test))
        return pred, clf.score(x_test,y_test)

