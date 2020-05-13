
# coding: utf-8

# In[ ]:

import pandas as pd

df = pd.read_csv('train_set.csv', sep='\t')

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

sw = STOPWORDS
# insert my_stopwords
sw.update(['one','say','first','second','new','two','will','us','also',            'said','U','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interp
from sklearn import svm
from sklearn.model_selection import GridSearchCV

X = df["Title"] + df["Content"]
df["Category"] = df["Category"].map({"Politics": 0, "Business": 1,"Football": 2,"Film": 3, "Technology": 4})
Y = df["Category"]

count_vect = CountVectorizer(stop_words=sw)
vectorizer = TfidfTransformer()

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
k = 10
kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
fold = 0

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
#metrics initialization
base_fpr = np.linspace(0, 1, 101)
fpr = tpr = thresholds = f1 = prc = rs = acc = auc =  0.0
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kf.split(X):
    
    X_train_counts = count_vect.fit_transform(X[train_index])
    X_train_counts = vectorizer.fit_transform(X_train_counts)
    svd = TruncatedSVD(n_components=200)
    X_lsi = svd.fit_transform(X_train_counts)
    
    parameters = {'C':[1, 10]}
    svr = svm.LinearSVC()
    clf = GridSearchCV(svr, parameters)
    
    clf = clf.fit(X_lsi, Y[train_index])
    
    X_test_counts = count_vect.transform(X[test_index])
    X_test_counts = vectorizer.transform(X_test_counts)
    X_test_counts = svd.transform(X_test_counts)
    yPred = clf.predict(X_test_counts)
    
    fold += 1
    
    # Calculating metrics
    f1 += f1_score(yPred,Y[test_index],average="macro")
    prc += precision_score(yPred,Y[test_index],average="macro")
    rs += recall_score(yPred,Y[test_index],average="macro")
    acc += accuracy_score(yPred, Y[test_index])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yPred,Y[test_index], pos_label = 2)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)
    auc += roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= kf.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

#roc_10fold
roc10 = np.zeros((4,3),dtype=object) #saving values for roc_10fold plot 
roc10[0,0] = mean_fpr
roc10[0,1] = mean_tpr
roc10[0,2] = mean_auc

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC_linear classifier\'s roc plot')
plt.legend(loc="lower right")
plt.show() 
    
print "Average f-score : " + str(f1/k) 
print "Average precision : " + str(prc/k)
print "Average recall_score : " + str(rs/k)
print "Average Accuracy : " + str(acc/k) 
print "Average AUC : " + str(auc/k)

em = np.zeros((4,5)) #evaluation metric matrix
em[0,0] = f1/k
em[0,1] = prc/k
em[0,2] = rs/k
em[0,3] = acc/k
em[0,4] = auc/k


# In[ ]:

from sklearn.naive_bayes import GaussianNB

kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
fold = 0
fpr = tpr = thresholds = f1 = prc = rs = acc = auc =  0
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kf.split(X):
    
    X_train_counts = count_vect.fit_transform(X[train_index])
    X_train_counts = vectorizer.fit_transform(X_train_counts)
    svd = TruncatedSVD(n_components=200)
    X_lsi = svd.fit_transform(X_train_counts)
    
    clf = GaussianNB()
    
    clf = clf.fit(X_lsi, Y[train_index])
    
    X_test_counts = count_vect.transform(X[test_index])
    X_test_counts = vectorizer.transform(X_test_counts)
    X_test_counts = svd.transform(X_test_counts)
    yPred = clf.predict(X_test_counts)
    
    fold += 1
    
    # Calculating metrics
    f1 += f1_score(yPred,Y[test_index],average="macro")
    prc += precision_score(yPred,Y[test_index],average="macro")
    rs += recall_score(yPred,Y[test_index],average="macro")
    acc += accuracy_score(yPred, Y[test_index])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yPred,Y[test_index], pos_label = 2)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)
    auc += roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr /= kf.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

roc10[1,0] = mean_fpr
roc10[1,1] = mean_tpr
roc10[1,2] = mean_auc

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GaussianNB classifier\'s roc plot')
plt.legend(loc="lower right")
plt.show()
 
print "Average f-score : " + str(f1/k) 
print "Average precision : " + str(prc/k)
print "Average recall_score : " + str(rs/k)
print "Average Accuracy : " + str(acc/k) 
print "Average AUC : " + str(auc/k)
em[1,0] = f1/k
em[1,1] = prc/k
em[1,2] = rs/k
em[1,3] = acc/k
em[1,4] = auc/k


# In[ ]:

from sklearn.ensemble import RandomForestClassifier

kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
fold = 0
fpr = tpr = thresholds = f1 = prc = rs = acc = auc =  0

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kf.split(X):
    
    X_train_counts = count_vect.fit_transform(X[train_index])
    X_train_counts = vectorizer.fit_transform(X_train_counts)
    svd = TruncatedSVD(n_components=200)
    X_lsi = svd.fit_transform(X_train_counts)
    
    clf = RandomForestClassifier(n_estimators = 20 ,n_jobs = -1)
    
    clf = clf.fit(X_lsi, Y[train_index])
    
    X_test_counts = count_vect.transform(X[test_index])
    X_test_counts = vectorizer.transform(X_test_counts)
    X_test_counts = svd.transform(X_test_counts)
    yPred = clf.predict(X_test_counts)
    
    fold += 1
    
    # Calculating metrics
    f1 += f1_score(yPred,Y[test_index],average="macro")
    prc += precision_score(yPred,Y[test_index],average="macro")
    rs += recall_score(yPred,Y[test_index],average="macro")
    acc += accuracy_score(yPred, Y[test_index])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yPred,Y[test_index], pos_label = 2)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)
    auc += roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= kf.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

roc10[2,0] = mean_fpr
roc10[2,1] = mean_tpr
roc10[2,2] = mean_auc
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest classifier\'s roc plot')
plt.legend(loc="lower right")
plt.show() 
 
print "Average f-score : " + str(f1/k) 
print "Average precision : " + str(prc/k)
print "Average recall_score : " + str(rs/k)
print "Average Accuracy : " + str(acc/k) 
print "Average AUC : " + str(auc/k)
          
em[2,0] = f1/k
em[2,1] = prc/k
em[2,2] = rs/k
em[2,3] = acc/k
em[2,4] = auc/k


# In[ ]:

import math
def euclideanDistance(instance1, instance2):
    mylist1=[]
    mylist2=[]
    for i in instance1:
        mylist1.append(i)
    for j in instance2:
        mylist2.append(j)
    points = zip(mylist1,mylist2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))

import operator

def tuple_distance(training_instance, test_instance,val):
    return (val, euclideanDistance(test_instance, training_instance))

def getKneighbors(trainingSet, testInstance, k):
    distances = [tuple_distance(training_instance, testInstance,val) for val,training_instance in enumerate(trainingSet)]
    # index 1 is the calculated distance between training_instance and test_instance
    distances.sort(key=operator.itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in distances]
    # select first k elements
    return sorted_training_instances[:k]

from collections import Counter

def getResponse(neighbors):
    classes = [neighbour[4] for neighbour in neighbors]
    cnt=Counter()
    for x in classes:
        if x in cnt:
            cnt[x] +=1
        else:
            cnt[x] = 1
    majority_sorted = sorted(cnt.items(),key=operator.itemgetter(0),reverse=True)
    return majority_sorted[0][0] 

kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
fold = 0
fpr = tpr = thresholds = f1 = prc = rs = acc = auc =  0

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kf.split(X):
    
    X_train_counts = count_vect.fit_transform(X[train_index])
    X_train_counts = vectorizer.fit_transform(X_train_counts)
    svd = TruncatedSVD(n_components=200)
    X_lsi = svd.fit_transform(X_train_counts)
    
    X_test_counts = count_vect.transform(X[test_index])
    X_test_counts = vectorizer.transform(X_test_counts)
    X_test_counts = svd.transform(X_test_counts)
    ypred=[]
    neighlist=[]
    A = np.array(df)
    mycount=0
    for i in X_test_counts:
        neighbors=getKneighbors(X_lsi,i,10)
        neighlist=[A[j,:] for j in neighbors]
        myres = getResponse(neighlist)
        ypred.append(myres)
        neighlist=[]
        print ypred
        mycount += 1
    
    fold += 1
    print str(fold)
    # Calculating metrics
    f1 += f1_score(yPred,Y[test_index],average="macro")
    prc += precision_score(yPred,Y[test_index],average="macro")
    rs += recall_score(yPred,Y[test_index],average="macro")
    acc += accuracy_score(yPred, Y[test_index])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yPred,Y[test_index], pos_label = 2)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)
    auc += roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr /= kf.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

roc10[3,0] = mean_fpr
roc10[3,1] = mean_tpr
roc10[3,2] = mean_auc

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN classifier\'s roc plot')
plt.legend(loc="lower right")
plt.show()
 
print( "Average f-score : " + str(f1/k)) 
print( "Average precision : " + str(prc/k))
print( "Average recall_score : " + str(rs/k))
print( "Average Accuracy : " + str(acc/k) )
print( "Average AUC : " + str(auc/k))
em[3,0] = f1/k
em[3,1] = prc/k
em[3,2] = rs/k
em[3,3] = acc/k
em[3,4] = auc/k


# In[ ]:

#Evaluation metric
rows = ["SVC_linear","GaussianNB","RandomForest","KNN"]
columns = ["F-measure","Precision","Recall_score","Accuracy","AUC"]
mycsv = pd.DataFrame(data=em,index=rows,columns=columns)
mycsv.to_csv("EvaluationMetric_10fold.csv")
print "Created EvaluationMetric_10fold.csv file!"

#roc_10fold
plt.plot(roc10[0,0], roc10[0,1], 'k--',
         label='SVClinear Mean_ROC (area = %0.2f)' % roc10[0,2], lw=2)
plt.plot(roc10[1,0], roc10[1,1], 'k--',
         label='GNB Mean_ROC (area = %0.2f)' %  roc10[1,2], lw=2)
plt.plot(roc10[2,0], roc10[2,1], 'k--',
         label='RF Mean_ROC (area = %0.2f)' %  roc10[2,2], lw=2)
plt.plot(roc10[3,0], roc10[3,1], 'k--',
         label='KNN Mean_ROC (area = %0.2f)' %  roc10[3,2], lw=2)
        
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classifiers\'s mean_roc plot')
plt.legend(loc="lower right")
plt.savefig("roc_10fold.png")
plt.show() 
print "Created roc_10fold.png file!"


# In[ ]:



