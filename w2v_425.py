import pyprind
import pandas as pd
import os
import re
import numpy as np
import gensim
import gensim.models
from sklearn import svm

# Collect stopwords

# stop = stopwords.words('english')
stop=[]
with open('stopwords.txt','r') as f:
	for line in f:
		stop.append(line)

# Transfer word to word vector

def tokenizer(text):#把一个小文档拆成词语转化为向量
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

basepath = 'data'
label = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
data = [] #加入第一行？？？
labels=[]

for s in ('test', 'train'):#看子文件夹
    for l in ('pos', 'neg'):#看子文件夹
        path = os.path.join(basepath, s, l)
        label_1=label[l]#生成其中一个txt的路径
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:#打开这个txt
                txt = infile.read()
                token = tokenizer(txt)#转化成向量
                data.append([token])
                labels.append(label_1)
                #, ignore_index=True)#加入csv文件中
            pbar.update()
#print(data[1])
#print(labels)

with open("data.txt", 'w', encoding='utf-8') as f:#保存成txt
    for i in data:
        for j in i:
            f.write(str(j))
        f.write('\n')

with open("labels.txt", 'w', encoding='utf-8') as f:#保存成txt
    for i in range(len(labels)):
        f.write(str(labels[i]))
        # f.write('\n') DO NOT enable this line, it will make error.

# Training word vector

inpath = 'data.txt'
outpath = 'w2v_trainResult'
pbar = pyprind.ProgBar(100000)

class csvStream(object):
    def __init__(self,path):
        self.path=path
    def __iter__(self): #把text弄成了一个list？？
        f = open(self.path,"r")
        with open(self.path, 'r',) as f:
            lines = f.readlines()
            for line in lines:
                #print('1:',line)
                text = line[1:-2] #只保留词语，前后都有3/4个不是词语的东东
                text = re.sub('[\'\"\[\]\d\b]','',text)   #替换了一些符号（？）
                while (text[0] == ',') or (text[0] == ' '):
                    text = text[1:]
                #print('2:',text)
                text=text.split(', ')
                #print('3:',text)
                yield text
                pbar.update()

lineIterator = csvStream(inpath)
print('start training ...')
train_model = gensim.models.Word2Vec(lineIterator, vector_size=200, window=5, min_count=5, workers=4, epochs=10) # 
train_model.save(outpath)
print('training complete.')

# Use SVM to predict sentiment

train_inpath = 'w2v_trainResult'
data='data.txt'
label='labels.txt'

model = gensim.models.Word2Vec.load(train_inpath)

vecs=[]
labels_train=[]
labels_test=[]
tests=[]
labels=[]

pbar = pyprind.ProgBar(50000)

def get_word_vector(path):
    ip = open(path, 'r', encoding='utf-8')
    content = ip.readlines()
    for i in range(50000):
        count=0
        vec=[0]*200
        line=content[i]
        line=eval(line)
        for word in line:
            try:
                count += 1
                vec+=model.wv[word]
            except KeyError:
                continue
        vec /= count
        #vec=np.array(vec).reshape([1,-1])
        vecs.append(vec)
        pbar.update()
    # print(vecs[0]) check vector


def get_label(path):
    ip = open(path, 'r', encoding='utf-8')
    content=ip.readlines()
    for i in range(len(content[0])):
        labels.append(int(content[0][i]))
    
def get_test_vector(path):
    ip = open(path, 'r', encoding='utf-8')
    content = ip.readlines()
    for i in range(25000,50000):
        vec=[0]*200
        line=content[i][1:-5]
        line=line.split(',')
        for i in range(len(line)):
            line[i]=line[i][2:-1]
        count = 0
        for word in line:
            try:
                count += 1
                vec+=model.wv[word]
            except KeyError:
                continue
        vec /= count
        tests.append(vec)

print('\nGet vector...')
get_label(label)
get_word_vector(data)
print('Vector get.')

clf = svm.SVC()  # class 
clf.fit(vecs[0:25000], labels[0:25000])
print('\nok')

pbar = pyprind.ProgBar(50000)
print('\nCounting accuracy...')
correct=0
for i in range(25000,50000):
    test=np.array(vecs[i]).reshape([1,-1])
    result = clf.predict(test)
    if result==labels[i]:
        correct+=1
    pbar.update()
print('\naccuracy=',correct/25000)