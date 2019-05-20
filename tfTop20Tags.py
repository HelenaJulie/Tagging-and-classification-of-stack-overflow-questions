# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:03:59 2019

@author: helen
"""

import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
import string
from string import punctuation

#
              
class tf():
    def __init__ (self):
        df = pd.read_csv('Questions.csv')
        df1 = pd.read_csv('Tags.csv')
        self.df2=pd.read_csv('Test1.csv')
        #        df=df.drop(columns=['Id'])
        #        print(df)
        
        self.train_questions= np.zeros((24000,4))
        self.train_questions=np.array(df)[:24000,:4]



        for i in range(0,200):
            self.train_questions[i][3]=''
            self.train_questions[i][0]=int(self.train_questions[i][0])


        train_tags=np.zeros((27000,2))
        train_tags=np.array(df1)[27000,:2]
        self.train_tags_new=np.array(df1)[:27000,1]
            
        train_tags.tolist()
        self.train_questions.tolist()
        for i in range(0,27000):
            for j in range(0,24000):
                if (train_tags[i][0] == self.train_questions[j][0]):
                    x=self.train_questions[j][3]+' '+train_tags[i][1]
                    self.train_questions[j][3]=x  
            
            

##         
    def finduniquetags(self):
        self.unique_tags = []
        self.unique_tagsCount=[]
        for i in range(0,len(self.train_questions)):
            tags = []  
            tags=self.train_questions[i][3].split()
            for j in tags:
                if j not in self.unique_tags:
                    self.unique_tags.append(j)
                    self.unique_tagsCount.append(0)
                    
        for i in range(0,len(self.train_questions)):
            tags = []  
            tags=self.train_questions[i][3].split()
            for j in tags:
                pos=self.unique_tags.index(j)
                self.unique_tagsCount[pos]=self.unique_tagsCount[pos]+1
        print(self.unique_tagsCount)
        

        
        #print(self.unique_tags.shape)
        
    def findtop20tag(self):
          
        index=sorted(range(len(self.unique_tagsCount)),key=self.unique_tagsCount.__getitem__)
        self.top20tags=[]
        self.top20tagcount=[]
        for i in range(len(index),len(index)-20):
            pos=index[i]
            self.top20tags.append(self.unique_tags[pos])
            self.top20tagcount.append(self.unique_tagsCount[pos])
        for i in range(0,len(self.top20tags)):
            print(self.top20tags[i],self.top20tagcount[i])
        print(self.top20tags)

t=tf()
t.finduniquetags()
t.findtop20tag()
