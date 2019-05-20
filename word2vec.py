#reference:http://nadbordrozd.github.io/posts/2/
#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

import os
import pandas as pd
import nltk
import numpy as np
import gensim
from sklearn.base import BaseEstimator, TransformerMixin
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.sklearn_api import W2VTransformer
nltk.download('punkt')
import pandas as pd
import nltk
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import hamming_loss, recall_score, jaccard_similarity_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
df = pd.read_csv('C:/Users/ebin1/Desktop/mine.csv');

df['Tokens'] = df['Tokens'].str[2:]
corpus = df['Tokens'].values.tolist()
#print(corpus)# every sentence is an
df["Tags"] = df['Tags'].str[4:].str.replace('[^\w\s]','')
tags=df["Tags"]
tag_corp = [nltk.word_tokenize(sent) for sent in tags]
#print(tag_corp)

alltags=[]


alltags.extend([item for sublist in tag_corp for item in sublist])
testtags=list(set(alltags))


tok_corp = [nltk.word_tokenize(sent) for sent in corpus]
print("done tokenizing")


#print(tok_corp)
model = gensim.models.Word2Vec(tok_corp, min_count=2, size=256, workers=4,sg=1,iter=100)
print(model.most_similar('number'))
print("_-------------------------")
model.save('testmodel')
model = gensim.models.Word2Vec.load('testmodel')

# print(model.wv.syn0)
# print("-----")
# print(model.wv.index2word)
# print("------")
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
# print("------")

class TfidfEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.w2v = w2v
        self.word2weight = None
        self.dim = 256

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.w2v[w] * self.word2weight[w]
                         for w in words if w in self.w2v] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


class AverageEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.w2v = w2v

        #dimensionality of our vector is 256
        self.dim =256

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.sum([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



from sklearn.pipeline import Pipeline
mlb = MultiLabelBinarizer()

X=corpus
Y = mlb.fit_transform(tag_corp)
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1,random_state=random_state)

pipe1=Pipeline([("wordVectz",AverageEmbeddingVectorizer(w2v)),
                ("multilabel",OneVsRestClassifier(LinearSVC()))])

pipe2=Pipeline([("wordVectz",TfidfEmbeddingVectorizer(w2v)),
                ("multilabel",OneVsRestClassifier(LinearSVC()))])
pipe1.fit(X_train, y_train)
predicted = pipe1.predict(X_test)
all_labels = mlb.inverse_transform(predicted)
print(all_labels)


accuracy1 = accuracy_score(y_test, predicted)
print("accuracy=",accuracy1)
#
print("Evaluation- BOW, SVC")
precision1 = precision_score(y_test, predicted,average='macro')
print(precision1)

ham_loss=hamming_loss(y_test,predicted)
print("hamming loss=",ham_loss)
recall1=recall_score(y_test,predicted,average='macro')
print("recall=",recall1)
j1=jaccard_similarity_score(y_test,predicted)
print("jaccard similarity score=",j1)
f1=f1_score(y_test, predicted,average='macro')
print("f1 score=",f1)
accuracy1 = accuracy_score(y_test, predicted)
print("accuracy=",accuracy1)
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform(tag_corp)
#
#
# random_state = np.random.RandomState(0)
# X=corpus
# # Split into training and test
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1,
#                                                     random_state=random_state)
#
#
#
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', OneVsRestClassifier(LinearSVC()))])
#
# classifier.fit(X_train, y_train)
# predicted = classifier.predict(X_test)
# all_labels = mlb.transform(predicted)#alllabes= all our predictions
# print(all_labels)
# print(y_test.shape)
# print(predicted.shape)
# print(all_labels)
# # for item, labels in zip(X_test, all_labels):
# #     print('{0} => {1}\n'.format(item, ', '.join(labels)))

