import sqlite3 as lite
import os.path 
from fuzzywuzzy import fuzz


class DbWorks(object):
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), 'dictionaries/all.db')
        if not os.path.exists(self.db_path):
            raise Exception("database path wrong/file not found"
            .format(os.path.abspath(self.db_path)))
            
        self.con = lite.connect(self.db_path)
            
        

    def search_string(self,string):

        with self.con:
            cur = self.con.cursor()    
            cur.execute("SELECT * FROM dictionary where word like ?",(string,))
            row = cur.fetchone()
            
            return row
            
            
    def levenschtein_similarity(self,string):
        with self.con:
            cur = self.con.cursor()
            cur.execute("SELECT * FROM dictionary where word like ? ",('%'+string[:3]+'%',))
            rows = cur.fetchall()
            result = []
            if not rows:
                return None
            
            else:         
                maxm = 0
                counter = 0                
                for i,row in enumerate(rows):
                    a=fuzz.ratio(string, row[1])
                    if a > maxm :
                        maxm = a
                        counter = i
                        
                if maxm>75:
                    result.append(rows[counter][0])
                    result.append(string)
                    result.append(rows[counter][2])
                    result.append(rows[counter][3])
                    
                    return result
                else:
                    return None
            
    


def main():
    db = DbWorks()
    #t  = db.search_string("others")
    t = db.levenschtein_similarity("eyes.")
    print t
    
 
if __name__ == "__main__":
        
    main()
    

    
    
    
    