# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 01:04:01 2016

@author: faisal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 01:01:19 2016

@author: faisal
"""


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
from sklearn.cross_validation import train_test_split


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
feature = np.array(dataset_features)
#labels
labels = list(set(X))
label = np.array([labels.index(x) for x in X])

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.