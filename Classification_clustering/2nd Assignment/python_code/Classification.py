import pandas as pd
import numpy as np

dataframe = pd.read_csv('train.tsv', sep='\t')
Y = dataframe['Label']

dataframe = dataframe.drop('Label',axis = 1)
dataframe = dataframe.drop('Id',axis = 1)

dataX = pd.get_dummies(dataframe)
X = dataX.as_matrix()

from scipy import interp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score

## SVM

k = 10
kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
acc = 0

for train_index, test_index in kf.split(dataX):
    
    #parameters = {'C':[1, 10]}
    clf = svm.LinearSVC()
    #clf = GridSearchCV(svr, parameters)
        
    clf = clf.fit(X[train_index], Y[train_index])    
    yPred = clf.predict(X[test_index])

    # Calculating accuracy
    acc += accuracy_score(yPred, Y[test_index])
    

em = np.zeros((1,3)) #storing mean accuracy for EvaluationMetric.csv

em[0,0] = acc/k
print ( "Average Accuracy of SVC_linear : " + str(acc/k) )


from sklearn.ensemble import RandomForestClassifier

kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
acc = 0

for train_index, test_index in kf.split(dataX):
    
    clf = RandomForestClassifier(n_estimators = 20 ,n_jobs = -1)
    
    clf = clf.fit(X[train_index], Y[train_index])
    
    yPred = clf.predict(X[test_index])

    # Calculating accuracy
    acc += accuracy_score(yPred, Y[test_index])
    

em[0,1] = acc/k
print ("Average Accuracy of RandomForest : " + str(acc/k) )

from sklearn.naive_bayes import GaussianNB

kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
acc = 0
for train_index, test_index in kf.split(X):
    
    clf = GaussianNB()
    
    clf = clf.fit(X[train_index], Y[train_index])
    yPred = clf.predict(X[test_index])

    # Calculating metrics
    acc += accuracy_score(yPred, Y[test_index])

print("Average Accuracy GaussianNB: " + str(acc/k) )
em[0,2] = acc/k


#Evaluation metric
columns = ["SVM","RandomForest","GaussianNB"]
rows = ["Accuracy"]
mycsv = pd.DataFrame(data=em,index=rows,columns=columns)
mycsv.to_csv("EvaluationMetric_10fold.csv")
print ("Created EvaluationMetric_10fold.csv file!")


test = pd.read_csv('test.tsv', sep='\t')

clf = RandomForestClassifier(n_estimators = 20 ,n_jobs = -1)
clf = clf.fit(X, Y)
yPred = clf.predict(X)

testX = test.drop('Id',axis = 1)
testX = pd.get_dummies(testX)

predicted = clf.predict(testX)

output = np.zeros((len(predicted),2),dtype=object)

for i,j in zip(predicted,range(len(predicted))):
    if i == 1:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Good"
    elif i == 2:
        output[j][0] = test.iloc[j]["Id"]
        output[j][1] = "Bad"

mycsv = pd.DataFrame(data=output,columns=["Client_ID","Predicted_Label"])
mycsv.to_csv("testSet_categories.csv", sep='\t',index=False, header=False)
print ("Created testSet_categories.csv file!")
