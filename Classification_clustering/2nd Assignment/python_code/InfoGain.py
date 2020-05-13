import math
def my_entropy(data):
    goods=0
    Y=data['Label']
    for i in Y:
        if i==1:
            goods+=1
    if data.shape[0]==0:
        return 0
    else:
        p_good= goods/data.shape[0]
    p_bad= 1 - p_good
    s1=-(p_good * math.log2(p_good)) 
    s2=-(p_bad * math.log2(p_bad))
    s_fin = s1+s2
    return s_fin

def info_sum_attribute(arg):
    templist=list(set(df.columns))
    nonused_cols=[v for v in templist if not v.startswith('Label')]
    nonused_cols.remove(arg)
    newdf=df.drop(nonused_cols,axis=1)
    sum=0
    myvals=newdf[arg].value_counts()
    mycount=0
    for i in myvals.index:
        df_temp=newdf[newdf[arg]==i]
        temp2=my_entropy(df_temp)
        amount_to_sum=(myvals[mycount]/newdf.shape[0]) * temp2
        sum+= amount_to_sum
        mycount+=1
    return sum

def info_num_sum(arg):
    templist=list(set(df.columns))
    nonused_cols=[v for v in templist if not v.startswith('Label')]
    nonused_cols.remove(arg)
    newdf=df.drop(nonused_cols,axis=1)
    sum=0
    (counters,bins)=np.histogram(newdf[arg],5)
    print(counters,bins)
    mycount=0
    j=0
    for i in counters:
        if j<4:
            df_temp=newdf[(newdf[arg]>=bins[j]) & (newdf[arg]<bins[j+1])]
        else:
            df_temp=newdf[(newdf[arg]>=bins[j]) & (newdf[arg]<=bins[j+1])]
        j+=1
        temp2=my_entropy(df_temp)
        amount_to_sum=(i/newdf.shape[0]) * temp2
        sum+= amount_to_sum
    return sum

info_list=[]

print("Entropy : "+ str(my_entropy(df)))
cols_to_drop=[v for v in cols_to_drop if not v.startswith('Label')]
catdata=df.drop(cols_to_drop,axis=1)
loopdata=catdata.drop('Label',axis=1)
for i in loopdata.columns:
  infogain=my_entropy(df)-info_sum_attribute(i)
  info_list.append([infogain,i])

cols_to_transform=[v for v in cols_to_transform if not v.startswith('Label')]
numericdata=df.drop(cols_to_transform,axis=1)
loopdata=numericdata.drop('Label',axis=1)
cols = []
print("Counters & bins")
for i in loopdata.columns:
    infogain=my_entropy(df)-info_num_sum(i)
    info_list.append([infogain,i])
    cols.append(i)
print()
print(info_list)
