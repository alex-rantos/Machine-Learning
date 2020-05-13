
# coding: utf-8

# In[1]:

import pandas as pd

df = pd.read_csv('train_set.csv', sep='\t')

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

sw = STOPWORDS
# insert my_stopwords
sw.update(['one','say','first','second','new','two','will','us','also', 
	'said','U','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


# In[2]:

technology = df[df["Category"] == "Technology"]
wordcloud_technology = WordCloud(width=600,height=400,stopwords=sw).generate(' '.join(technology["Content"]))
plt.figure()
plt.imshow(wordcloud_technology, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[3]:

business = df[df["Category"] == "Business"]
wordcloud_business = WordCloud(width=600,height=400,stopwords=sw).generate(' '.join(business["Content"]))
plt.figure()
plt.imshow(wordcloud_business, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[4]:

film = df[df["Category"] == "Film"]
wordcloud_film = WordCloud(width=600,height=400,stopwords=sw).generate(' '.join(film["Content"]))
plt.figure()
plt.imshow(wordcloud_film, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[5]:

football = df[df["Category"] == "Football"]
wordcloud_football = WordCloud(width=600,height=400,stopwords=sw).generate(' '.join(football["Content"]))
plt.figure()
plt.imshow(wordcloud_football, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[6]:

politics = df[df["Category"] == "Politics"]
wordcloud_politics = WordCloud(width=600,height=400,stopwords=sw).generate(' '.join(politics["Content"]))
plt.figure()
plt.imshow(wordcloud_politics, interpolation="bilinear")
plt.axis("off")
plt.show()

