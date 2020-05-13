#info_list sorted based on info_gain
import operator
accs=[]
info_list=sorted(info_list,key=operator.itemgetter(0))

df = pd.read_csv('train.tsv', sep='\t')
tls=['Label','Id']
df=df.drop(tls,axis=1)


for col in info_list:
    if len(df.columns)<2:
        break
    df = df.drop(col[1],axis=1)
    dataX = pd.get_dummies(df)
    X = dataX.as_matrix()
    
    k = 10
    kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=0)
    acc = 0
    
    for train_index, test_index in kf.split(dataX):
        clf = RandomForestClassifier(n_estimators = 20 ,n_jobs = -1)
        clf = clf.fit(X[train_index], Y[train_index])
        yPred = clf.predict(X[test_index])
        acc += accuracy_score(yPred, Y[test_index])

    mean_acc = acc/k
    accs.append(mean_acc)

    
info_df=pd.DataFrame(info_list)
info_df.to_csv('FeatureSelection+InfoGain.csv')    
# Plot configurations
from numpy  import array
a = np.zeros((len(info_list)),dtype=object)
counter = 0
for col in info_list:
    a[counter] = col[1]
    counter += 1  
plt.xticks(range(len(a)),a,size='medium',rotation='vertical')
plt.ylabel('Mean Accuracy')
plt.xlabel('Feature')
plt.plot(accs)

fig1 = plt.gcf()
plt.show()
plt.draw()

fig1.savefig('FeatureSelection')


# In[16]:

print(info_list)
tmp=[ ]   

df = pd.read_csv('train.tsv', sep='\t')
Y = df['Label']
df = df.drop('Label',axis=1)
df = df.drop('Id',axis=1)
for i in info_list:
    tmp.append(i[1])

if compare_value < max(accs):
    n = accs.index(max(accs))
    cols_to_drop = tmp[:n+1]
    print(cols_to_drop)
    df = df.drop(cols_to_drop,axis=1)
    test = pd.read_csv('test.tsv', sep='\t')
    test = test.drop(cols_to_drop,axis=1)
    
    
    dataX = pd.get_dummies(df)
    X = dataX.as_matrix()
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
    mycsv.to_csv("DropFeature_testSet_categories.csv", sep='\t',index=False, header=False)
    print ("Created testSet_categories.csv file!")


