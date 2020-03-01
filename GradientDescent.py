from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

def MSE(pred,x):
    error = pred - x
    loss = error**2
    loss = np.sum(loss)/pred.shape[0]
    #We are passing error as large value will  create problem in convergence#
    return [error,loss]

epoch=100
alpha=0.01
X, y = make_blobs(n_samples=10, n_features=2,random_state=0)
'''make_blobs is used to make isotropic blobs based on gaussian for clustering. This is used as classifier
n_samples ---> Total rows 
n_features --> Total columns'''
trainX,testX,trainY,testY = train_test_split(X,y)


W = np.random.randn(X.shape[1])
b = np.ones((trainX.shape[0]))

#Extra#
loss = 1000.0
epoch = 0
loss_log = []
epoch_log = []
while loss>0.099 and epoch<500:
    y = trainX.dot(W)+b #y=wx+b#
    
    error,loss = MSE(y,trainY)
    print("total epoch",epoch,"total loss",loss)
    m = trainX.shape[0]
    #See derivative formula#
    temp0 = b - (alpha*error)/m
    temp1 = W - (alpha*trainX.T.dot(error))/m
    #Note if it is divided by m then epoch is increased#
    #Because learning rate becomes very low 
    b = temp0
    W = temp1
    loss_log.append(loss)
    epoch_log.append(epoch)
    epoch+=1

'''Testing'''
pred = testX.dot(W)
pred = 1.0/1+np.exp(-pred) #Sigmoid#
for i in  range(0,pred.shape[0]):
    if pred[i]>0.5:
        pred[i] = 1
    else:
        pred[i] = 0
    if testY[i]>0.5:
        testY[i] = 1
    else:
        testY[i] = 0
print(confusion_matrix(pred,testY))
print(accuracy_score(pred,testY))
