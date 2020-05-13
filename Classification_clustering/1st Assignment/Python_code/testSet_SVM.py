
# coding: utf-8

# In[6]:

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interp
from sklearn import svm
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('train_set.csv', sep='\t')

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
k = 10

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

sw = STOPWORDS
# insert my_stopwords
sw.update(['one','say','first','second','new','two','will','us','also',            'said','U','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
X = df["Title"] + df["Content"]
df["Category"] = df["Category"].map({"Politics": 0, "Business": 1,"Football": 2,"Film": 3, "Technology": 4})
Y = df["Category"]

count_vect = CountVectorizer(stop_words=sw)
vectorizer = TfidfTransformer()


# In[7]:

test = pd.read_csv('test_set.csv', sep='\t')
testX = test["Title"] + test["Content"]
    
X_train_counts = count_vect.fit_transform(X)
X_train_counts = vectorizer.fit_transform(X_train_counts)
svd = TruncatedSVD(n_components=200)
X_lsi = svd.fit_transform(X_train_counts)

parameters = {'C':[1, 10]}
svr = svm.LinearSVC()
clf = GridSearchCV(svr, parameters)

clf = clf.fit(X_lsi, Y)

X_test_counts = count_vect.transform(testX)
X_test_counts = vectorizer.transform(X_test_counts)
X_test_counts = svd.transform(X_test_counts)
predicted = clf.predict(X_test_counts)

output = np.zeros((len(predicted),2),dtype=object)

for i,j in zip(predicted,range(len(predicted))):
    if i == 0:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Politics"
    elif i == 1:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Business"
    elif i == 2:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Football"
    elif i == 3:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Film"
    elif i == 4:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Technology"

mycsv = pd.DataFrame(data=output,columns=["ID","Predicted_Category"])
mycsv.to_csv("testSet_categories.csv", sep='\t',index=False, header=False)
print "Created testSet_categories.csv file!"


# In[ ]:



