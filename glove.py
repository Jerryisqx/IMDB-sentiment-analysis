import pandas as pd
import os
import gensim
from gensim import corpora
import pyprind
from string import punctuation
import numpy as np
from mittens import GloVe
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown

class Glove(object):
    def __init__(self,pos_comment_train=None,pos_comment_test=None,neg_comment_train=None,\
        neg_comment_test=None,data=None, dict=None,token_id=None,n_matrix=None,window=None,\
            word_matrix=None,n_dims=None):
        self.pos_comment_train=pos_comment_train
        self.pos_comment_test=pos_comment_test
        self.neg_comment_train=neg_comment_train
        self.neg_comment_test=neg_comment_test
        self.data=data
        self.dict = corpora.Dictionary(data)
        self.token_id = dict.token2id
        self.n_matrix = len(token_id)
        self.window = 5
        self.word_matrix = np.zeros(shape=[n_matrix, n_matrix])
        self.n_dims = 100
        self.labels = None
        self.vecs = None


    def open_file(self):
        self.pos_comment_train = open('pos_review_train.txt','r').readlines()
        self.pos_comment_test = open('pos_review_test.txt','r').readlines()
        self.neg_comment_train = open('neg_review_train.txt','r').readlines()
        self.neg_comment_test = open('neg_review_test.txt','r').readlines()
        print('okkkk')

    def preprocessing(self,text,text2):
        with open(text,'w') as f:
            for i in text2:
                text_list = nltk.word_tokenize(i)
                english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','<','/','..','|']
                text_list = [word for word in text_list if word not in english_punctuations]
                stops = set(stopwords.words("english"))
                text_list = [word for word in text_list if word not in stops]
                YAM = np.array(nltk.pos_tag(text_list))
                a = ((YAM[:,1] == 'JJ'))
                selected = YAM[a][:,0]
                alist = selected.tolist()
                if len(alist) < 5:
                    selected = str(alist)
                    f.write(''.join(selected)[1:len(''.join(selected))-1])
                    f.write('\n')
                else:
                    alist = alist[(len(alist) - 5):len(alist)]
                    selected = str(alist)
                    f.write(''.join(selected)[1:len(''.join(selected))-1])
                    f.write('\n')
        print('hiiii')
    def get_data(self):
        self.preprocessing('1.txt',self.pos_comment_train)
        self.preprocessing('2.txt',self.pos_comment_test)
        self.preprocessing('3.txt',self.neg_comment_train)
        self.preprocessing('4.txt',self.neg_comment_test)
        f1=open("1.txt","r",encoding="utf-8").read().splitlines()
        f2=open("2.txt","r",encoding="utf-8").read().splitlines()
        f3=open("3.txt","r",encoding="utf-8").read().splitlines()
        f4=open("4.txt","r",encoding="utf-8").read().splitlines()
        self.data=f1+f2+f3+f4
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        #remove blank samples due to cutting down words 
        for i in range(0,13367):
            data[i]=list(eval(data[i]))
        for i in range(13368,22098):
            data[i]=list(eval(data[i]))
        for i in range(22099,23957):
            data[i]=list(eval(data[i]))
        for i in range(23958,25264):
            data[i]=list(eval(data[i]))
        for i in range(25265,27170):
            data[i]=list(eval(data[i]))
        for i in range(27171,34438):
            data[i]=list(eval(data[i]))
        for i in range(34439,34550):
            data[i]=list(eval(data[i]))
        for i in range(34551,36666):
            data[i]=list(eval(data[i]))
        for i in range(36667,38064):
            data[i]=list(eval(data[i]))
        for i in range(38065,39466):
            data[i]=list(eval(data[i]))
        for i in range(39467,42002):
            data[i]=list(eval(data[i]))
        for i in range(42003,42214):
            data[i]=list(eval(data[i]))
        for i in range(42215,49087):
            data[i]=list(eval(data[i]))
        for i in range(49088,50000):
            data[i]=list(eval(data[i]))
        del data[13367]
        del data[22097]
        del data[23955]
        del data[25261]
        del data[27166]
        del data[34433]
        del data[34544]
        del data[36659]
        del data[38056]
        del data[39457]
        del data[41992]
        del data[42203]
        del data[49075]
        g=open("data.txt","w+",encoding="utf-8")
        for i in range(len(data)):
            string=",".join(data[i])
            g.write(string)
            g.write("\n")
        g.close()

    def form_labels(self):
        label=["1"]*12500+["0"]*12497+["1"]*12495+["0"]*12495
        h=open("labels.txt","w+",encoding="utf-8")
        for i in range(len(label)):
            h.write(label[i])
        h.close()
    
    def Bottom_Top(self,c_pos, max_len, window):
        bottom = c_pos - window
        top = c_pos + window + 1
        if bottom < 0:
            bottom = 0
        if top >= max_len:
            top = max_len
        return bottom, top
       
    def form_wordmatrix(self):
        for i in range(len(data)):
            k = len(self.data[i])
            for j in range(k):
                bottom, top = Bottom_Top(j, k, window)
                c_word = data[i][j]
                c_pos = token_id[c_word]
                for m in range(bottom, top):
                    t_word = data[i][m]
                    if m != j and t_word != c_word:
                        t_pos = token_id[t_word]
                        word_matrix[c_pos][t_pos] += 1

    def form_Glove(self):
        glove = GloVe(n=n_dims, max_iter=100, learning_rate=0.005)
        G = glove.fit(word_matrix)

    def form_mittens(self):
        items_index=[]
        for keys,items in dict.iteritems():
            items_index.append(items)
        out_str=[]
        for i in range(len(token_id)):
            s=items_index[i]
            for j in range(n_dims):
                s=s+" "+str(G[i][j])
            out_str.append(s)

        #form mittens.txt
        f=open("mittens.txt","w+",encoding="utf-8")
        s=str(len(token_id))+" "+str(n_dims)+"\n"
        f.write(s)
        for i in out_str:
            f.write(i)
            f.write("\n")
        f.close()
        #save mittens.txt as word2vec form
        model = gensim.models.KeyedVectors.load_word2vec_format("mittens.txt")
        model.word_vec

        #use feature vector to represent each text
        data='data.txt'
        label='new_labels.txt'
        model = gensim.models.KeyedVectors.load_word2vec_format("mittens.txt",binary=False)
        vecs=[]
        labels_train=[]
        labels_test=[]
        tests=[]
        labels=[]
        pbar = pyprind.ProgBar(50000)
        def get_word_vector(path):
            ip = open(path, 'r', encoding='utf-8')
            content = ip.readlines()
            for i in range(len(content)):
                count=0
                vec=[0]*100
                line=content[i]
                line=eval(line)
                for word in line:
                    try:
                        count += 1
                        vec+=model[word]
                    except KeyError:
                        continue
                vec=np.divide(vec,count)
                vecs.append(vec)
                pbar.update()

        def get_label(path):
            ip = open(path, 'r', encoding='utf-8')
            content=ip.readlines()
            for i in range(len(content[0])):
                labels.append(int(content[0][i]))

        get_label(label)
        get_word_vector(data)
        self.labels = labels
        self.vecs = vecs




g=Glove()
g.open_file()
g.get_data()
g.formlabels()
g.form_wordmatrix()
g.form_Glove()
g.form_mittens()