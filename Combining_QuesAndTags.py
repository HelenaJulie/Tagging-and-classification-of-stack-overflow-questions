# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:58:35 2019

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
#df = pd.read_csv('Questions.csv')
#df1 = pd.read_csv('Tags.csv')
#df2=pd.read_csv('Test1.csv')
##df=df.drop(columns=['Id'])
#print(df)
#train_questions= np.zeros((200,4))
#train_questions=np.array(df)[:200,:4]
#for i in range(0,200):
#    train_questions[i][3]=''
#    train_questions[i][0]=int(train_questions[i][0])
#
import csv
file = open("train.csv",encoding="utf8")
df = csv.reader(file)
out_file = open("new12.csv","w",newline='',encoding="utf8")
writer = csv.writer(out_file)
for row in df:
    s=re.sub("[^a-zA-Z0-9,]", "", row[1])
    st=re.sub("[,]"," ",s)
    
    row[1]=st
    stt=re.sub("[^a-zA-Z0-9']", " ", row[0]).lower()
    row[0]=stt
    if(row[0]!="" and row[1]!=""):
        writer.writerow(row)
 

        
    
file.close()
    
out_file.close()
        
