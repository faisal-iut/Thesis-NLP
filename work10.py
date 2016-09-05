# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 02:53:46 2016

@author: faisal
"""

import os
import json
import pickle
import nltk
from collections import defaultdict
from pprint import pprint
import json
import work3
import csv
#==============================================================================
# datadict = defaultdict(list)
# splitter = nltk.data.load('tokenizers/punkt/english.pickle')
# f = open("Data.txt")
# i=1
# for lines in f.readlines():
#     if lines!="\n":
#         print i,lines
#         sentences = splitter.tokenize(lines)
#         tag= sentences[len(sentences)-1]
#         del sentences[len(sentences)-1]
#         print sentences
#         lines = ' '.join(sentences)
#         print lines
#         i=i+1
#         datadict[tag].append(lines)
#         
# pprint (datadict)
# 
# # Open a file for writing
# out_file = open("data.json","w")
# 
# # Save the dictionary into this file
# # (the 'indent=4' is optional, but makes it more readable)
# json.dump(datadict,out_file, indent=4)                                    
# 
# # Close the file
# out_file.close()             
#==============================================================================
with open('data4.json') as data_file:    
    data = json.load(data_file)

wholeset =[]
for pera in data.values():
     new = work3.match(str(pera))
     wholeset.append(new)

with open("out4.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv)
    writer.writerow(['word', 'pos','prepos','nxtpos','ontotag','onto-BIO','onto-predict','dict-tag','dict-score','dict-BIO'])
    for new1 in wholeset:    
        for sentence in new1:
            for word in sentence:
                writer.writerow([word[0],word[1],word[2],word[3],word[4],word[5],word[6],word[7],word[8],word[9]])



