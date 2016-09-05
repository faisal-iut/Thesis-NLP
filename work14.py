# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:14:55 2016

@author: faisal
"""

from __future__ import division

import csv
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from pprint import pprint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    conf_mat2 = np.around(cm,decimals=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    for x in tick_marks:
        for y in tick_marks:
            plt.annotate(str(conf_mat2[x][y]),xy=(y,x),ha='center',va='center')
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png',dpi=900,orientation='portrait', papertype=None, format=None,
        transparent=True)


estimator = SVC(kernel='linear')
gammas = np.logspace(-6, -1, 10)

dataset = defaultdict(list)  

with open('out1.csv', 'rb') as mycsv1:
    dictofdata = csv.DictReader(mycsv1)
    data1 = [row for row in dictofdata]  
with open('out4.csv', 'rb') as mycsv2:
    dictofdata = csv.DictReader(mycsv2)
    data2 = [row for row in dictofdata]

#dataset    
dataset = data1+data2
X = [row['tag'] for row in dataset]

for i,row in enumerate(dataset):
    del dataset[i]['word']
    del dataset[i]['tag']
    del dataset[i]['bio']
    del dataset[i]['onto-BIO']
    del dataset[i]['dict-BIO']
#    del dataset[i]['dict-tag']
#    del dataset[i]['pos']
#    del dataset[i]['prepos']
#    del dataset[i]['nxtpos']
#    del dataset[i]['onto-predict']
#    del dataset[i]['dict-score']
#    del dataset[i]['ontotag']
#print dataset    
dataset_features = []
labels = []
label = []
feature = []
#features
vec = DictVectorizer()
dataset_features= vec.fit_transform(dataset).toarray()
feature = np.array(dataset_features)
#labels
labels = list(set(X))
label = np.array([labels.index(x) for x in X])

n_samples, n_features = feature.shape 
p = range(n_samples) # an index array, 0:n_samples
random.seed(random.random())
random.shuffle(p) # the index array is now shuffled

feature, label = feature[p], label[p] # both the arrays are now shuffled

kfold = 5 # no. of folds (better to have this at the start of the code)
c = np.zeros((7,7))
skf = StratifiedKFold(label,kfold,shuffle=False)
print skf
skfind = [None]*len(skf) # indices
cnt=0
for train_index in skf:
  skfind[cnt] = train_index
  cnt = cnt + 1

acc = 0
res = []
res1 = []
for i in range(kfold):
    train_indices = skfind[i][0]
    test_indices = skfind[i][1]
    clf = []
    clf = svm.LinearSVC(C=1)
    clf1 = DummyClassifier(strategy='stratified',random_state=0)
    clf2 = LogisticRegression(random_state=1)
    clf3 = RandomForestClassifier(random_state=1)
    clf4 = GaussianNB()
    clf0 = VotingClassifier(estimators=[('lr', clf2), ('rf', clf3), ('gnb', clf4)], voting='hard')
    X_train = feature[train_indices]
    y_train = label[train_indices]
    X_test = feature[test_indices]
    y_test = label[test_indices]
     # Training

    clf.fit(X_train,y_train)
    clf1.fit(X_train,y_train) 
    
    y_predict = []
    y_predict1 = []

    y_predict = clf.predict(X_test) # output is labels and not indices
    y_predict1 = clf1.predict(X_test) # output is labels and not indices
    
    sc= clf.score(X_test,y_test)
    sc1= clf1.score(X_test,y_test)
    cl1 = classification_report(y_test, y_predict, target_names=labels, digits=3 )
    cl2 = classification_report(y_test, y_predict1, target_names=labels, digits=3)
    print cl1    
    lines = cl1.split('\n')
    lines1 = cl2.split('\n')
    classes = []
    support = []
    class_names = []
    classes1 = []
    support1 = []
    class_names1 = []
#==============================================================================
#     for line in lines[2 : (len(lines) - 2)]:
#         t = line.strip().split()
#         if len(t) < 2: continue
#         classes.append(t[0])
#         v = [float(x) for x in t[1: len(t) - 1]]
#         support.append(int(t[-1]))
#         class_names.append(t[0])
#         print(class_names)
#==============================================================================
    v=[]
    v1=[]     
    print lines[10]
    t = lines[10].strip().split()
    t1 = lines1[10].strip().split()
    v.append(i+1)
    v1.append(i+1)    
    for x in t[3: len(t) - 1]:
        v.append(float(x))
    v.append(int(t[-1]))       
    for x in t1[3: len(t1) - 1]:
        v1.append(float(x))
    v1.append(int(t1[-1]))       

    v.append(float("{0:.3f}".format(sc)))
    v1.append(float("{0:.3f}".format(sc1)))
    print v
    res.append(v)
    res1.append(v1)

        
    cm = confusion_matrix(y_test, y_predict)
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    c = c+cm
    acc = acc+sc
    print sc,sc1

print res1
#==============================================================================
# with open("result-with all-base.csv", 'wb') as outcsv:
#         writer = csv.writer(outcsv)
#         writer.writerow(['iteration','precision','recall','f1-score','support','accuracy'])
#         for word in res1:
#             writer.writerow([word[0],word[1],word[2],word[3],word[4],word[5]])
#==============================================================================
avg_acc = acc/5
cm_normalized = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
print sum(sum(c)),sum(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()


    
    
    

    

 

