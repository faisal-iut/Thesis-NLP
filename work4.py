
from __future__ import division

import csv
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn import svm
from pprint import pprint
import pickle
#==============================================================================
# csv.register_dialect(
#     'mydialect',
#     delimiter = ',',
#     quotechar = '"',
#     doublequote = True,
#     skipinitialspace = True,
#     lineterminator = '\r\n',
#     quoting = csv.QUOTE_MINIMAL)
#==============================================================================
    
#==============================================================================
# print("\n Now the output from a dictionary created from the csv file")
# symplist=[]
# with open('codes\work\dictionaries\symp-lemma.csv', 'rb') as mycsvfile:
#     dictofdata1 = csv.DictReader(mycsvfile, dialect='mydialect')
#     for row in dictofdata1:
#         f= row['symp-lemma'].lower()
#         t = WordNetLemmatizer().lemmatize(f,'v')
#         if t not in symplist:
#             symplist.append(t)
#             print f,t
# print("\n Now the output from a dictionary created from the csv file")            
#==============================================================================
 
traindata = defaultdict(list)  
with open('out1.csv', 'rb') as mycsv1:
    dictofdata = csv.DictReader(mycsv1)
    traindata = [row for row in dictofdata]

testdata = defaultdict(list)  
with open('out4.csv', 'rb') as mycsv2:
    dictofdata = csv.DictReader(mycsv2)
    testdata = [row for row in dictofdata]
    

    
X1 = [row['word'] for row in traindata]        
X2 = [row['word'] for row in testdata]


#==============================================================================

for i,row in enumerate(traindata):
    del traindata[i]['word']
    del traindata[i]['tag']



#==============================================================================

pprint (traindata)

vec = DictVectorizer()

train_features= vec.fit_transform(traindata).toarray()

train_features = np.array(train_features)


train_dataframe = pd.read_csv('output.csv')
train_labels = train_dataframe.tag
labels = list(set(train_labels))


     
train_labels = np.array([labels.index(x) for x in train_labels])

    
classifier = svm.LinearSVC()
classifier.fit(train_features, train_labels)

test_features = train_features
test_features = np.array(test_features)
test_labels = train_labels

results = classifier.predict(test_features)
print vec.get_feature_names()
for row in zip(train_features,train_labels,results):
    print row

num_correct = (results == test_labels).sum()

print num_correct,len(test_labels)

recall = num_correct / len(test_labels)
print recall*100
result = classifier.predict(test_features[3].reshape(1,-1))
print result

#==============================================================================
# with open('disease.csv', 'wb') as csvfile:
#     thedatawriter = csv.writer(csvfile)
#     for row in disease:
#         print row
#         thedatawriter.writerow([row])
#         
#==============================================================================
        
#==============================================================================
# with open('symptom.csv', 'wb') as csvfile:
#     thedatawriter = csv.writer(csvfile)
#     for row in symplist:
#         print row
#         thedatawriter.writerow([row])
# 
#==============================================================================


#==============================================================================
# with open("output.csv", 'wb') as outcsv:   
#     #configure writer to write standard csv file
#     writer = csv.writer(outcsv)
#     writer.writerow(['word', 'pos', 'ontotag','onto-BIO','onto-predict','dict-tag','dict-score','dict-BIO'])
#     for sentence in new:
#         for word in sentence:
#             writer.writerow([word[0],word[1],word[2],word[3],word[4],word[5],word[6],word[7]])
# 
#==============================================================================



