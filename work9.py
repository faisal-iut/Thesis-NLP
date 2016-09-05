# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:24:02 2016

@author: faisal
"""
import nltk
from sklearn.feature_extraction import  FeatureHasher

class Preprocess(object):
    
     def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
        
     def sentence_split(self,text):
        sentences = self.splitter.tokenize(text)
        return sentences

     def word_token(self,sentence):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        return tokenized_sentence
   
     def pos_tag(self, sentence):
        tagged = nltk.pos_tag(sentence)
            
        return tagged
        
        
        
     def process(self, text):
        splitted_sentences = self.sentence_split(text)
        #print splitted_sentences
        tokenized = [self.word_token(sentence) for sentence in splitted_sentences]
        #print "\n\n\n",tokenized
        pos_tagged = [self.pos_tag(sentence) for sentence in tokenized]
        #print "\n\n\n", pos_tagged 
        
        return pos_tagged 
        
     def token_features(self,token, part_of_speech):
        if token.isdigit():
            yield "numeric"
        else:
            yield "token={}".format(token.lower())
            yield "token,pos={},{}".format(token, part_of_speech)
        if token[0].isupper():
            yield "uppercase_initial"
        if token.isupper():
            yield "all_uppercase"
        yield "pos={}".format(part_of_speech)

def main():

    text = """I am fealing pain in the stomac for three-weeks. Sometimes
    I feel dizzy, weak and can't breathing_properly. I can't have rheumatic fever studying excessively, It pains barely."""

    preprocess = Preprocess()
    processed = preprocess.process(text)
   # processed_dict = defaultdict(list)
    raw=[]
    for sentence in processed:
        for word,pos in sentence:
            raw.append(preprocess.token_features(word,pos))
            
    print raw
    hasher = FeatureHasher(input_type='string')
    X = hasher.transform(raw)
    
    print X
    
    

if __name__ == "__main__":
    main()
