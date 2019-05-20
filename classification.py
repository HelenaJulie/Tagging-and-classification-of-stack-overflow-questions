import pandas as pd
import nltk
import numpy as np


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

nltk.download('punkt')

df = pd.read_csv('mine1.csv');

df['Tokens'] = df['Tokens'].str[2:]
corpus = df['Tokens'].values.tolist()
#print(corpus)# every sentence is an

df["Tags"] = df['Tags'].str[4:].str.replace('[^\w\s]','')
tags=df["Tags"]
tag_corp = [nltk.word_tokenize(sent) for sent in tags]
print(type(tag_corp))

tags1=pd.DataFrame(tag_corp) 
categories=list(tags1.columns.values)
alltags=[]


alltags.extend([item for sublist in tag_corp for item in sublist])

testtags=list(set(alltags))

tok_corp = [nltk.word_tokenize(sent) for sent in corpus]





mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tag_corp)


random_state = np.random.RandomState(0)
X=corpus
# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1,
                                                    random_state=random_state)


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)#alllabes= all our predictions

print(predicted)
print(y_test)
for item, labels in zip(X_test, all_labels):
    print('{0} => {1}\n'.format(item, ', '.join(labels)))
 
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


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', BinaryRelevance(GaussianNB()))])
    
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)#alllabes= all our predictions
for item, labels in zip(X_test, all_labels):
    print('{0} => {1}\n'.format(item, ', '.join(labels)))
print("Evaluation- BOW, Naive bayes")
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

 


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LabelPowerset(LogisticRegression()))])

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)#alllabes= all our predictions
for item, labels in zip(X_test, all_labels):
    print('{0} => {1}\n'.format(item, ', '.join(labels)))
print("Evaluation- BOW, Logistic regression")
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


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLkNN(k=10))])

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)#alllabes= all our predictions
for item, labels in zip(X_test, all_labels):
    print('{0} => {1}\n'.format(item, ', '.join(labels)))
print("Evaluation- BOW, Multilabel KNN")
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
