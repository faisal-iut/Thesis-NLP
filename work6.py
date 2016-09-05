# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:11:33 2016

@author: faisal
"""
import re
from work5 import DbWorks
from work8 import Wordfeature
from collections import defaultdict
from work1 import Preprocess
from operator import itemgetter

class DictionaryTag(object):
    
    def __init__(self):
        self.db = DbWorks()
        self.wordfeature = Wordfeature()

    
    def tag(self,words):
        
        dict_tagged = defaultdict(list)
        for string in words:
            if len(string) > 2: 
                temp = self.db.search_string(string)
                #print temp
                split= re.findall(r'\S+', string)
                if temp is None and len(split)==1:
                    temp = self.db.levenschtein_similarity(string)
                if temp is None:
                    temp = self.wordfeature.is_weight(string)
                    if temp is None:     
                        temp = self.wordfeature.is_measurement(string)
                        if temp is None:
                            temp = self.wordfeature.is_volume(string)
                if temp is not None:
                    split_phrase= re.findall(r'\S+', temp[1])
                    for count,word in enumerate(split_phrase):
                        if len(word)>1:
                            if word not in dict_tagged:
                                if count == 0:
                                    dict_tagged[word].append((temp[2],temp[3],'B'))
                                else:
                                    dict_tagged[word].append((temp[2],temp[3],'I'))
                            
        return dict_tagged


if __name__ == "__main__":
    text = """Diagnosed with crohn's in 2008. Very unwell for many years. 
    Tried different meds but untolerated/ no difference. Occasional back pain but nothing 
    like since i started infliximab about 6 months ago. Consultant requested a break for few weeks
    to see if pain subsides- not a lot of difference. Had another infusion this week, constant pain in 
    the back, worse at night, broken sleep due to pain. I just wondered if anyone has similar experience being on infliximab infusion. """    
    preprocess = Preprocess()
    dict_tagging = DictionaryTag()

    
    processed = preprocess.process(text)
    flat = [item for sublist in processed for item in sublist]
    listed = list(map(itemgetter(0),flat))
    flat_bi_tri = preprocess.bi_trigram(listed)
    tagged_dict = dict_tagging.tag(flat_bi_tri)
    print tagged_dict
