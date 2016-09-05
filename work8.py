# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:03:32 2016

@author: faisal
"""

import re

class Wordfeature(object):
    
    def __init__(self):
        self.d =1
        
        
    def is_size(self,word):
        """
        is_size()
    
        Purpose: Checks if the word is a size.
    
        @param word. A string.
        @return      the matched object if it is a weight, otheriwse None.
    
        >>> is_size('1mm') is not None
        True
        >>> is_size('10 cm') is not None
        True
        >>> is_size('36 millimeters') is not None
        True
        >>> is_size('423 centimeters') is not None
        True
        >>> is_size('328') is not None
        False
        >>> is_size('22 meters') is not None
        False
        >>> is_size('millimeters') is not None
        True
        """
        regex = r"^[0-9]*( )?(mm|cm|millimeters|centimeters)$"
        return re.search(regex, word)
        
    def is_weight(self,word):
        """
        is_weight()
    
        Purpose: Checks if word is a weight.
    
        @param word. A string.
        @return      the matched object if it is a weight, otherwise None.
    
        >>> is_weight('1mg') is not None
        True
        >>> is_weight('10 g') is not None
        True
        >>> is_weight('78 mcg') is not None
        True
        >>> is_weight('10000 milligrams') is not None
        True
        >>> is_weight('14 grams') is not None
        True
        >>> is_weight('-10 g') is not None
        False
        >>> is_weight('grams') is not None
        True
        """
              
        regex = r"^[0-9]*( )?(mg|g|mcg|milligrams|grams|pound|pounds|kg|kilogram|kilograms)$"
        temp = []          
        if re.search(regex, word):
            temp.append(0)
            temp.append(word)
            temp.append(4)
            temp.append(11)
            
            return temp
        else:
            return None
        
    def is_volume(self,word):
        """
        is_volume()
    
        Purpose: Checks if word is a volume.
    
        @param word. A string.
        @return      the matched object if it is a volume, otherwise None.
    
        >>> is_volume('9ml') is not None
        True
        >>> is_volume('10 mL') is not None
        True
        >>> is_volume('552 dL') is not None
        True
        >>> is_volume('73') is not None
        False
        >>> is_volume('ml') is not None
        True
        """
        regex = r"^[0-9]*( )?(ml|mL|dL)$"

        temp = []          
        if re.search(regex, word):
            temp.append(0)
            temp.append(word)
            temp.append(4)
            temp.append(12)
            
            return temp
        else:
            return None
        
        
    
    def is_measurement(self,word):
        """
        is_measurement()
    
        Purpose: Checks if the word is a measurement.
    
        @param word. A string.
        @return      the matched object if it is a measurement, otherwise None.
    
        >>> is_measurement('10units') is not None
        True
        >>> is_measurement('7 units') is not None
        True
        >>> is_measurement('10cc') is not None
        True
        >>> is_measurement('300 L') is not None
        True
        >>> is_measurement('20mL') is not None
        True
        >>> is_measurement('400000 dL') is not None
        True
        >>> is_measurement('30000') is not None
        False
        >>> is_measurement('20dl') is not None
        False
        >>> is_measurement('units') is not None
        True
        """
        regex = r"^[0-9]*( )?(unit(s)|cc|L|mL|dL|cm|mm|degree|%)$"
        temp = []          
        if re.search(regex, word):
            temp.append(0)
            temp.append(word)
            temp.append(4)
            temp.append(12)
            
            return temp
        else:
            return None
            
    def is_age(self,word):
        """
        is_measurement()
    
        Purpose: Checks if the word is a measurement.
    
        @param word. A string.
        @return      the matched object if it is a measurement, otherwise None.
    
        >>> is_measurement('10units') is not None
        True
        >>> is_measurement('7 units') is not None
        True
        >>> is_measurement('10cc') is not None
        True
        >>> is_measurement('300 L') is not None
        True
        >>> is_measurement('20mL') is not None
        True
        >>> is_measurement('400000 dL') is not None
        True
        >>> is_measurement('30000') is not None
        False
        >>> is_measurement('20dl') is not None
        False
        >>> is_measurement('units') is not None
        True
        """
        regex = r"(^[0-99]*( )?(years old))|(age is ^[0-99])$"
        temp = []          
        if re.search(regex, word):
            temp.append(0)
            temp.append(word)
            temp.append(4)
            temp.append(12)
            
            return temp
        else:
            return None
        

    def allfeatures(self,word):
        

        features = []    
        if wordfeature.is_size(word):
            features.append(1)
        else:
            features.append(0)
        
        if wordfeature.is_measurement(word):
            features.append(1)
        else:
            features.append(0)
            
        if wordfeature.is_volume(word):
            features.append(1)
        else:
            features.append(0)
            
        if wordfeature.is_weight(word):
            features.append(1)
        else:
            features.append(0)
            
        return features 
            
            
            
if __name__ == "__main__":
    wordfeature =  Wordfeature()    
    word = "22 cc"
    print wordfeature.is_age("age is 23")
    #print wordfeature.allfeatures(word) 
    
    

 
    