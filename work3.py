# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 21:11:23 2016

@author: faisal
"""


from work1 import Preprocess
from work2 import Ontotext
from work6 import DictionaryTag
from operator import itemgetter
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
import json
import pickle
from collections import defaultdict

def match(text):
    preprocess = Preprocess()
    ontotext =  Ontotext()
    dict_tagging = DictionaryTag()
    
    processed = preprocess.process(text)
    flat = [item for sublist in processed for item in sublist]
    listed = list(map(itemgetter(0),flat))
    flat_bi_tri = preprocess.bi_trigram(listed)
    tagged_dict = dict_tagging.tag(flat_bi_tri)

    joined = " ".join(listed)
    ontotagged =  ontotext.tag(joined)
    new = []
    for i,sentences in enumerate(processed):
        new.append(list())
        for j,word in enumerate(sentences):
            temp = processed[i][j][0]
            if j==0:
                prepos = "start"
            else:
                prepos = processed[i][j-1][1]
            if j==len(sentences)-1:
                nxtpos = "end"
            else:
                nxtpos = processed[i][j+1][1]
            #print pretag,word[1],nxttag
            if word[0] in ontotagged.keys():
                insert = [word[0],word[1],prepos,nxtpos,ontotagged[temp][0][0],ontotagged[temp][0][1],ontotagged[temp][0][2]]
                
            else:
                insert = [word[0],word[1],prepos,nxtpos,'NA','B','NA']
                
            if word[0] in tagged_dict.keys():
                insert.append(tagged_dict[temp][0][0])
                insert.append(tagged_dict[temp][0][1])
                insert.append(tagged_dict[temp][0][2])
            else:
                insert.append(0)
                insert.append(0)
                insert.append('B')

            new[i].append(tuple(insert))
    
    return new


def information_extraction(info):
    print info
    
def ml_interface(feature):

    clf = joblib.load('svc.pkl') 
    for sentence in feature:
        for word in sentence:
            print word
            
    with open("output.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
        writer = csv.writer(outcsv)
        writer.writerow(['word', 'pos','prepos','nxtpos','ontotag','onto-BIO','onto-predict','dict-tag','dict-score','dict-BIO'])
        for sentence in feature:
            for word in sentence:
                writer.writerow([word[0],word[1],word[2],word[3],word[4],word[5],word[6],word[7],word[8],word[9]])
                
    datafeature = defaultdict(list)  

    with open('output.csv', 'rb') as mycsv:
        dictofdata = csv.DictReader(mycsv)
        features = [row for row in dictofdata]
    
    for i,row in enumerate(features):
        del features[i]['word']
        del features[i]['onto-BIO']
        del features[i]['dict-BIO']
        
    vec = DictVectorizer()
    data_features= vec.fit_transform(features).toarray()
    data_features = np.array(data_features)
            
            

def main():
    
    text = """hello, i am 24 years old. for some days i have fever with moderate shaking chills. 
    sometimes the fever goes up to 106 degree. i sweat excessively. i have headache and nausea. I have also diarrhea and muscle pain."""
    
    new = match(text)

#==============================================================================
#     print new
#     
#     with open("input.pkl", 'wb') as f:
#         pickle.dump(new, f)
#     
#     with open("input.pkl", 'rb') as f:
#         new1 = pickle.load(f)
#         
#==============================================================================
    print new
    #ml_interface(new1)
    

if __name__ == "__main__":
    main()