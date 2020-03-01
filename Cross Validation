from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd

#reading data#
df = pd.read_csv("/Users/prituldave/Downloads/iris.data")
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
le = LabelEncoder()


#Model defining#
KNN = KNeighborsClassifier(5)

start_time = time.time()
#Leave One Out#
loo = LeaveOneOut()
splits = loo.get_n_splits(X,Y)

accuracy_score_list = []
for train_index,test_index in loo.split(X,Y):
    trainX,testX = X.iloc[train_index],X.iloc[test_index]
    trainY,testY = Y[train_index],Y[test_index] #iloc not required as we have only 1 feature in Y
    #Evaluating the model#
    le.fit(trainY)
    trainY = le.fit_transform(trainY)
    le.fit(testY)
    testY = le.fit_transform(testY)
    KNN.fit(trainX,trainY)
    y = KNN.predict(testX)
    score = accuracy_score(y,testY)
    accuracy_score_list.append(score)
end_time = time.time()
#Analyzing Leave One Out#
print("Leave One Out")
print("number of split is done",splits)
print("Inference Time",end_time - start_time)
print("Average score",sum(accuracy_score_list)/len(accuracy_score_list))
print(" ----------------------------------------------------------")


#K-Fold Cross Validation#
from sklearn.model_selection import cross_val_score
start_time = time.time()
le.fit(Y)
Y = le.transform(Y)
score = cross_val_score(estimator = KNN,X = X,y = Y,cv = 10)
end_time = time.time()

#Analysis#
print("10 Fold Cross Validation")
print("inference time ",end_time-start_time)
print("Average score",sum(score)/len(score))
print(" ----------------------------------------------------------")


#Stratified Cross Validation#
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
accuracy_score_list = []
for train_index,test_index in skf.split(X,Y):
    trainX,testX = X.iloc[train_index,:],X.iloc[test_index,:]
    trainY,testY = Y[train_index],Y[test_index] #iloc not required as we have only 1 feature in Y
    #Evaluating the model#
    le.fit(trainY)
    trainY = le.fit_transform(trainY)
    le.fit(testY)
    testY = le.fit_transform(testY)
    KNN.fit(trainX,trainY)
    y = KNN.predict(testX)
    score = accuracy_score(y,testY)
    accuracy_score_list.append(score)
end_time = time.time()
#Analyzing Leave One Out#
print("10 Stratified Cross Validation")
print("Inference Time",end_time - start_time)
print("Average score",sum(accuracy_score_list)/len(accuracy_score_list))

'''
Output:- 
Leave One Out
number of split is done 149
Inference Time 0.5387048721313477
Average score 0.3288590604026846
 ----------------------------------------------------------
10 Fold Cross Validation
inference time  0.02981090545654297
Average score 0.9666666666666668
 ----------------------------------------------------------
10 Stratified Cross Validation
Inference Time 0.057556867599487305
Average score 0.9666666666666668
'''


'''
Conclusion:-
Leave One Out has high variance as splitting is not done properly
Leave One Out taking highest time
K FOld and Stratified will nearly give same result
Among K Fold and Stratified, K fold takes lesser time
'''

