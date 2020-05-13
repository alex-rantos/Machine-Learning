
# coding: utf-8

# In[1]:

import pandas as pd

df = pd.read_csv('train_set.csv', sep='\t')

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

sw = STOPWORDS
# insert my_stopwords
sw.update(['one','say','first','second','new','two','will','us','also',            'said','U','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


# In[2]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

#Ignore Category column
df1 = df.drop(['Category'], axis=1)

count_vect = CountVectorizer(stop_words=sw)
X_train_counts = count_vect.fit_transform(df1.Content)
transformer = TfidfTransformer(smooth_idf=False)
X_train_counts=transformer.fit_transform(X_train_counts)
svd = TruncatedSVD(n_components=100)
X_train_counts=svd.fit_transform(X_train_counts)

import numpy as np
from sklearn.cluster import KMeans
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster import cosine_distance

# avoid dividing by 0
from numpy.linalg import norm
for v in X_train_counts:
    if norm(v,2) == 0:
        v += 1
        v /= norm(v,2)
        
NUM_CLUSTERS = 5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=cosine_distance, repeats=10)
assigned_clusters = kclusterer.cluster(X_train_counts, assign_clusters=True)
df["Category"] = df["Category"].map({"Politics": 0, "Business": 1,"Football": 2,"Film": 3, "Technology": 4})

df.head(5)


# In[3]:

A = np.array(df)
myarray = np.zeros((5,5))

from collections import Counter

#Cluster calculation by category 
cnt=Counter()
for i, val in enumerate(assigned_clusters):
    myarray[val,A[i,4]] += 1
    cnt[val] += 1
for i in range(myarray.shape[0]):
    for j in range(myarray.shape[1]):
        myarray[i,j] /= cnt[i]


# In[4]:

titles= ["Politics","Business","Football","Film","Technology"]
clusterpoints= ["Cluster1","Cluster2","Cluster3","Cluster4","Cluster5"]
mycsv = pd.DataFrame(data=myarray,index=clusterpoints,columns=titles)
mycsv.to_csv("clustering_KMeans.csv")


# In[ ]:



