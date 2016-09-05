# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:35:23 2016

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
from pprint import pprint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pprint import pprint

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def features(s):
    
    feature = {    
        'pos': s['pos'],
        'prepos': s['prepos'],
        'nxtpos': s['nxtpos'],        
    }
    
    return feature


with open('out1.csv', 'rb') as mycsv1:
    dictofdata = csv.DictReader(mycsv1)
    data1 = [row for row in dictofdata]  
with open('out4.csv', 'rb') as mycsv2:
    dictofdata = csv.DictReader(mycsv2)
    data2 = [row for row in dictofdata]

#dataset    
dataset = data1+data2
X = [row['bio'] for row in dataset]

for i,row in enumerate(dataset):
    del dataset[i]['word']
    del dataset[i]['bio']
    

s=3900
print len(dataset)
train_data = [dataset[:s]]
train_labels = [X[:s]]
test_data = [dataset[s:]]
test_labels = [X[s:]]
print len(train_data[0]),len(test_data[0])

#print train_data[0],train_labels[0]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)

crf.fit(train_data,train_labels)
y_pred = crf.predict(test_data)


labels = list(crf.classes_)
print labels
sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    test_labels, y_pred, labels=sorted_labels, digits=3
))

tlabels = list(set(X[s:]))
true_labels = np.array([labels.index(x) for x in X[s:]])
plabels = list(set(y_pred[0]))
pre_labels = np.array([labels.index(x) for x in y_pred[0]])

print len(true_labels),len(pre_labels) 

cm = confusion_matrix(true_labels, pre_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
