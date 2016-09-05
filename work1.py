# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:53:59 2016

@author: faisal
"""
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk import bigrams,trigrams
from collections import defaultdict
import re


replacement_patterns = [
  (r'won\'t', 'will not'),
  (r'can\'t', 'cannot'),
  (r'I\'m', 'i am'),
  (r'ain\'t', 'is not'),
  (r'(\w+)\'ll', '\g<1> will'),
  (r'(\w+)n\'t', '\g<1> not'),
  (r'(\w+)\'ve', '\g<1> have'),
  (r'(\w+)\'s', '\g<1> is'),
  (r'(\w+)\'re', '\g<1> are'),
  (r'(\w+)\'d', '\g<1> would'),
  (r'[-_]', ' '),
  (r'[(]|[)]|[,]|[:][.]', ''),
  (r'u\'', ''),
  (r'u\"', '')    
]


 
class Preprocess(object):
    def __init__(self,patterns=replacement_patterns):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def sentence_split(self,text):
        sentences = self.splitter.tokenize(text)
        return sentences

        
    def spell_check(self,sentence):
        checked = TextBlob(sentence).correct()
        checked_sentences = str(checked)
        return checked_sentences
        
    def word_token(self,sentence):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        return tokenized_sentence
        
    def lemmatizetion(self,tokenized_sentence):
        lemmatized_sentence = [self.lemmatizer.lemmatize(token,'v') for token in tokenized_sentence]
        return lemmatized_sentence
        
            
    def pos_tag(self, sentence):
        tagged = nltk.pos_tag(sentence)
            
        return tagged

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
          s = re.sub(pattern, repl, s)
        return s
        
    def bi_trigram(self,flat):
        words=[]
        bi_token = bigrams(flat)
        tri_token = trigrams(flat)
        bi = [" ".join(tuples) for tuples in bi_token]
        tri = [" ".join(tuples) for tuples in tri_token]
        
        
        words.append(tri)
        words.append(bi)
        words.append(flat)
        
        words = [item for sublist in words for item in sublist]
        
        return words
    
    def split_phrase(self,word):  
        
        split= re.findall(r'\S+', word)
        return split
        
        
    def process(self, text):
        text = self.replace(text)
        #print text
        splitted_sentences = self.sentence_split(text)
        #print splitted_sentences
        #spelled = [self.spell_check(sentence) for sentence in splitted_sentences]
        #print "\n\n\n", spelled
        low = [s.lower() for s in splitted_sentences]
        tokenized = [self.word_token(sentence) for sentence in low]
        #print "\n\n\n",tokenized
        lemmatized = [self.lemmatizetion(sentence) for sentence in tokenized ]
        #print "\n\n\n",lemmatized
        pos_tagged = [self.pos_tag(sentence) for sentence in lemmatized]
        #print "\n\n\n", pos_tagged 
        
        return pos_tagged

      
        
def main():

    text = """hello, i am 24 years old. for some days i have fever with moderate shaking chills. 
    sometimes the fever goes up to 106 degree. i sweat excessively. i have headache and nausea. I have also diarrhea and muscle pain."""

    preprocess = Preprocess()
    processed = preprocess.process(text)
   # processed_dict = defaultdict(list)
    
    print processed,text
    
    

if __name__ == "__main__":
    main()


    

