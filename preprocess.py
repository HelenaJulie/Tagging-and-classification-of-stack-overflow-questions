#This code is a sightly modified verison of  https://github.com/AAbercrombie0492/nlp_final_project/blob/master/code/preprocess.py 
from collections import Counter
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import _pickle as cPickle
import numpy as np
import pandas as pd
import re
import collections
import chardet
import nltk
nltk.download('wordnet')
nltk.download('stopwords')




#preprocessinig
class Prepocessing():

    def __init__(self, file='diy.csv', n_samples_per_file=1, rem_stopwords=True, xtite=True, ner=True,
                 rem_punc=True, lemmatize=True):
       
        
        self.file = file
        self.data = pd.DataFrame([])
        self.readf(n_samples_per_file)
        if xtite == True:
            self.xtite()
            self.concat_features(['triple_title', 'content'])
        else:
            self.concat_features(['title', 'content'])

        self.preprocess(rem_stopwords, xtite, ner, rem_punc, lemmatize)

    def readf(self, n_samples_each_file):
        
        df = pd.read_csv(self.file)
        self.data = pd.concat([self.data, df.iloc[:n_samples_each_file]])  # first 100 rows

    def xtite(self):
       
        self.data['triple_title'] = (' ' + self.data['title']) * 3

    def concat_features(self, features):
        

        baseline = self.data[features[0]]
        concatenated = baseline.str.cat('' + self.data[features[1]])
        self.data['ttl_ctxt'] = concatenated

    def preprocess(self, rem_stopwords=True, xtite=True, ner=True, rem_punc=True, lemmatize=True):
       

       
        self.data['tokens'] = self.data.loc[:, 'ttl_ctxt'].apply(
            lambda x: clean_data(x, rem_stopwords=rem_stopwords, ner=ner, rem_punc=rem_punc, lemmatize=lemmatize))

       
        self.data.loc[:, 'tags'] = self.data.loc[:, 'tags'].apply(lambda x: clean_text(x))

        
        self.word_counts = Counter(np.hstack(self.data.tokens.values))
        self.tag_counts = Counter(np.hstack(self.data.tags.values))

#remove stopwords
def clean_stopwords(text):
    
    tokens = text.split()
    output = []
    for t in tokens:
        if t not in stopwords.words():
            output.append(t)
    return ' '.join(output)

#removes all punctuations
def clean_punc(text):
   
    new_str = re.sub(r'[^\w\s]', ' ', text)
    new_str = re.sub(r'(\n)', '', new_str)
    new_str = re.sub(r'(\\t)', '', new_str)
    new_str = re.sub(r'(\\x)', '', new_str)
    return new_str

#lammatizes words
def stem_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    output = []
    for t in tokens:
        lemmanade = wordnet_lemmatizer.lemmatize(t)
        lemmanade = wordnet_lemmatizer.lemmatize(lemmanade, pos='v')
        output.append(lemmanade)

  
    return ' '.join(output)

def clean_data(x, rem_stopwords=True, ner=True, rem_punc=True, lemmatize=False):
    
    x = x.encode('UTF-8') 
    icode = chardet.detect(bytes(x))['encoding']  
    try:
        decoded_s = x.decode(icode)  
    except:
        decoded_s = x.decode('UTF-8')
    re.sub(r'\w{4}', u'xxxx', decoded_s, flags=re.UNICODE)  
    x = decoded_s.encode('UTF-8')  
    clean_str = str(x)  
    if rem_stopwords == True:
        clean_str= clean_stopwords(clean_str)

    if lemmatize == True:
        clean_str = stem_text(clean_str)

    soup = BeautifulSoup(clean_str, 'html.parser')
    clean_str = soup.text

    if rem_punc == True:
        clean_str = clean_punc(clean_str)


    return clean_str

#changes caracter encodings
def clean_text(text_string):


    text_string = text_string.encode('UTF-8') 
    icode = chardet.detect(bytes(text_string))['encoding']  
    decoded_s = text_string.decode(icode)  
    re.sub(r'\w{4}', u'xxxx', decoded_s, flags=re.UNICODE)  
    text_string = decoded_s.encode('UTF-8')  
    text_string = str(text_string)  

    text_string = re.sub(r'[^\w\s]', ' ', text_string)
    text_string = re.sub(r'[\[\]\"\'\,]', ' ', text_string)  
    text_string = text_string.split()  
    return text_string


#main
if __name__ == '__main__':
    main_obj = Prepocessing(file='diy.csv', n_samples_per_file=200, rem_stopwords=True, xtitle=True,
                                   ner=True, rem_punc=True, lemmatize=True)
    main_obj.data.to_csv("Dataset.csv", index=True, header=False)#output of cleaned dataset with tags and tokens


