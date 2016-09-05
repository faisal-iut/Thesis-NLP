# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 01:27:02 2016

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
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

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
    
#features
vec = DictVectorizer()
dataset_features= vec.fit_transform(dataset).toarray()
dataset_features = np.array(dataset_features)
#labels
labels = list(set(X))
dataset_labels = np.array([labels.index(x) for x in X])
#traindata
s = 3900
train_data = dataset[:s]
train_features = dataset_features[:s]
train_labels = dataset_labels[:s]
#testdata
test_data = dataset[s:]
test_features = dataset_features[s:]
test_labels = dataset_labels[s:]

print len(train_data),len(train_features),len(train_labels),len(test_data),len(test_features),len(test_labels),len(dataset[0]),len(train_features[0])

#train classifier
lr = LogisticRegression()
gnb = GaussianNB()
rfc = RandomForestClassifier(n_estimators=100)
svc = svm.LinearSVC(C=1)

#==============================================================================
# lr.fit(train_features, train_labels)
# gnb.fit(train_features, train_labels)
# rfc.fit(train_features, train_labels)
#==============================================================================
svc.fit(train_features, train_labels)
#dump model
#==============================================================================
# joblib.dump(lr, 'lr.pkl')
# joblib.dump(gnb, 'gnb.pkl')
# joblib.dump(rfc, 'rfc.pkl')
#==============================================================================
#joblib.dump(svc, 'svc.pkl')
#load model
#clf = joblib.load('svc.pkl') 
#result
results = svc.predict(test_features)
num_correct = (results == test_labels).sum()
print num_correct,len(test_labels)
y_true = test_labels
y_pred = results
accuracy = accuracy_score(test_labels, results)*100
print accuracy, svc.score(test_features, test_labels)
print(classification_report(y_true, y_pred, target_names=labels))




cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(sum(sum(cm_normalized)))
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
print_cm(cm,labels)




