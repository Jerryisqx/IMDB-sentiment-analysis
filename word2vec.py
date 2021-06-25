import pandas as pd
import os
import re
import numpy as np
import gensim
import gensim.models

class Word2Vec(object):
    def __init__(self,stopwords=[],basepath= 'data',label = {'pos': 1, 'neg': 0},data=[],labels=[],model=None):
        self.stopwords=stopwords
        self.basepath=basepath
        self.label = label
        self.data = data 
        self.labels=labels
        self.model=model

    def get_stopwords(self):
        stop=[]
        with open('stopwords.txt','r') as f:
            for line in f:
                self.stopwords.append(line)

    def tokenizer(self,text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower()) +\
            ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in text.split() if w not in self.stopwords]
        return tokenized

    def preprocessing(self):
        for s in ('train', 'test'):#看子文件夹
            for l in ('pos', 'neg'):#看子文件夹
                path = os.path.join(self.basepath, s, l)
                label1=self.label[l]#生成其中一个txt的路径
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:#打开这个txt
                        txt = infile.read()
                        token = self.tokenizer(txt)#转化成向量
                        self.data.append(token)
                        self.labels.append(label1)
     
    def form_word2vec(self):
        self.get_stopwords()
        self.preprocessing()
        self.model = gensim.models.Word2Vec()
        self.model.build_vocab(self.data)
        return self.data,self.labels, self.model
