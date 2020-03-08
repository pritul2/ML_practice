'''
This is the implementation of Breast Cancer detection using Multi Layer Perceptron to understand how MLP works
The dataset used is of sklearn dataset breast cancer
The obtained accuracy is 87%
'''
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

df = sklearn.datasets.load_breast_cancer()


x = df.data # 30 features#
y = df.target #0 or 1#

#Binarizing the input for MP neuron#
x = pd.DataFrame(x)
x = x.apply(pd.cut,bins=2,labels=[1,0]) #From data we infered that 1,0 working better
x = np.array(x)
            
            
trainX,testX,trainY,testY = train_test_split(x,y,stratify=y) #maintaining the ratio#

#Obtaining best threshold value using brute force#
thresh=0
valid_cnt = 0
prev = 0
best_thresh = 0
while thresh<trainX.shape[0]+1:
    valid_cnt = 0
    for x,y in zip(trainX,trainY):
        if np.sum(x)>=thresh:
            ypred = 1
        else:
            ypred = 0
        if ypred == y:
            valid_cnt +=1
    acc = valid_cnt/trainX.shape[0]
    print("accuracy",acc)
    thresh+=1
    if acc>prev:
        prev = acc
        best_thresh = thresh
        
print("best threshold value is  ",best_thresh) #28#


#Testing the dataset using sklearn accuracy_score#
pred_array = []
for x in testX:
    if np.sum(x)>=best_thresh:
        pred = 1
    else:
        pred = 0
    pred_array.append(pred)
print("accuracy at testing dataset ",accuracy_score(testY,pred_array)) #0.8741258741258742#
    
