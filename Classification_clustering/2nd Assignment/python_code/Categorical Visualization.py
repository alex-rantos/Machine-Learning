import pandas as pd
import pylab
df = pd.read_csv('train.tsv', sep='\t')

from os import path
import matplotlib.pyplot as plt

cols_to_transform = list(set(df.columns)-set(df._get_numeric_data().columns))
new_cols=[]
for col in cols_to_transform:
    col += ','
    new_cols.append(col)

good = df[df["Label"]==1]
bad = df[df["Label"]==2]
    
cols_to_drop = df._get_numeric_data().columns
good_cat = good.drop( cols_to_drop, axis = 1 )
good_dict = good_cat.to_dict(orient='records')
bad_cat = bad.drop( cols_to_drop, axis = 1 )
bad_dict = bad_cat.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer as DV

vectorizer = DV(separator=',',sparse=False)
goodvec = vectorizer.fit_transform(good_dict)
badvec = vectorizer.transform(bad_dict)
good2 = pd.DataFrame(goodvec,columns=vectorizer.get_feature_names())
bad2 = pd.DataFrame(badvec,columns=vectorizer.get_feature_names())
for single_col in new_cols:
    filter_col=[col for col in list(vectorizer.get_feature_names()) if col.startswith(single_col)]
    good3=good2[filter_col]
    bad3 = bad2[filter_col]
    from collections import Counter
    cnt = Counter()
    cnt2 = Counter()
    for index,row in good3.iterrows():
        for i in filter_col:
         if(row[i]==1.0):
          cnt[i]+=1
    for index,row in bad3.iterrows():
        for i in filter_col:
         if(row[i]==1.0):
          cnt2[i]+=1
    import numpy as np
    l = np.arange(len(cnt.keys()))
    width=0.25
    fig = plt.figure(figsize=(2 * len(cnt.keys()), 4))
    ax = fig.add_subplot(111)
    bars1 = ax.bar(l, cnt.values(), width,linewidth=2* len(cnt.keys()) * width,align='center',color='b')
    plt.xticks(l,cnt.keys())
    bars2 = ax.bar(l + width, cnt2.values(),width,linewidth =2 * len(cnt.keys()) * width, align='center',color='r')
    plt.xticks(l, cnt2.keys())
    ax.legend( (bars1[0], bars2[0]), ('Good', 'Bad') )
    plt.show()
    fig.savefig('Categorical Visualization %s' % single_col)
