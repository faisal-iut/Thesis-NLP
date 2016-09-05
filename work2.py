# Copyright  2013, 2014, Ontotext AD
#
# This file is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

# IMPORTANT!!! Please install the packages below, needed for this example

import requests
import json
import yaml
import re
from collections import defaultdict


class Ontotext(object):
    def __init__(self):
        self.endpoint = "https://text.s4.ontotext.com/v1/sbt"
        self.api_key = "s43vbvdlp5ba"
        self.key_secret = "th9s5mv52dhtppp"
        stream = open('../dictionaries/ontotext.yml', 'r')
        self.entityname = yaml.load(stream)
         
    def tag(self,text):
        data = {
            "document": text,
            "documentType": "text/plain"
        }
        jsonData = json.dumps(data)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        }
        
        req = requests.post(
            self.endpoint, headers=headers,
            data=jsonData, auth=(self.api_key , self.key_secret))
            
        response = json.loads(req.content.decode("utf-8"))
        
        tag_result = defaultdict(list)
        for key in response["entities"]:
            for index,item in enumerate(response["entities"][key]):
                #if key in entityname.keys():
                    #print(entityname[key])
                temp = response["entities"][key][index]["string"]
                if key in self.entityname.keys():
                    predict = self.entityname[key]
                else:
                    predict = "NA"
                split_phrase= re.findall(r'\S+', temp)
                for count,word in enumerate(split_phrase):
                    if word not in tag_result.keys():
                        if count == 0:
                            tag_result[word].append((key,'B',predict))
                        else:
                            tag_result[word].append((key,'I',predict))
                
                
        return tag_result

    


def main():

    text = """I am fealing pain in the stomac for three-weeks. Sometimes
    I feel dizzy, weak and can't breathing_properly. I can't have rheumatic fever studying excessively, It pains barely."""
    
    ontotext =  Ontotext()
    t =  ontotext.tag(text)
    print t




if __name__ == "__main__":
    main()