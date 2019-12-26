#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import *
from math import log, exp
import numpy as np
from numpy.random import random
import time
import cvxopt, cvxopt.solvers


# In[ ]:


test_data= pd.read_csv("test_data.txt", delimiter=" ",header=None)
test_label= pd.read_csv("test_label.txt",delimiter=" ",header=None)
train_data= pd.read_csv("train_data.txt", delimiter=" ",header=None)
train_label= pd.read_csv("train_label.txt", delimiter=" ",header=None)


# In[ ]:


# train_data_mean = pd.DataFrame(data=train_data)
mean_value= train_data.mean(axis=0)


# In[ ]:


# train_data_std = pd.DataFrame(data=train_data)
std_dev=train_data.std(axis=0)


# In[ ]:


train_data_normalised=(train_data-mean_value)/std_dev


# In[ ]:


test_data_normalised=(test_data-mean_value)/std_dev


# In[ ]:


def fit(train_data, train_label, C):
  n,m = train_data.shape
  Q = np.zeros((m+n+1,m+n+1))
  for i in range(m):
      Q[i,i] = 1
  c = np.vstack([np.zeros((m+1,1)), C*np.ones((n,1))])
  A = np.zeros((2*n, m+1+n))
  A[:n,0:m] = train_label*train_data
  A[:n,m] = train_label.T
  A[:n,m+1:]  = np.eye(n)
  A[n:,m+1:] = np.eye(n)
  A = -A
  g = np.zeros((2*n,1))
  g[:n] = -1
  ## E and d are not used in the primal form
  ## convert to array
  ## have to convert everything to cxvopt matrices
  Q = cvxopt.matrix(Q,Q.shape,'d')
  c = cvxopt.matrix(c,c.shape,'d')
  A = cvxopt.matrix(A,A.shape,'d')
  g = cvxopt.matrix(g,g.shape,'d')
  ## set up cvxopt
  ## z (the vector being minimized for) in this case is [w, b, eps].T
  sol = cvxopt.solvers.qp(Q, c, A, g)
  return sol
     


# In[ ]:


train_data_normalised=train_data_normalised.to_numpy()


# In[ ]:


train_label=train_label.to_numpy()


# 

# In[ ]:


def train_svm(train_data1,train_label,val_data,val_label,C=0.25):
    p = fit(train_data1,train_label,C)
    wP = np.array(p['x'][:train_data1.shape[1]])
    bP = p['x'][train_data1.shape[1]]
    xValidate = test_data1.to_numpy()
   
    yP = xValidate.dot(wP) + bP
    
    a=yP
    a[a>0]=1
    a[a<0]=-1
    return accuracy_score(a, test_label.to_numpy())


# In[ ]:


def k_FoldCrossValidation(X, y, k = 5,C=0.25):
    
    numberOfTrainingExamples = y.size
    # Run Validation k times, with n/k elements in each division
    numberOfElements = (numberOfTrainingExamples // k)
    all_acc=[]
    print(X.shape)
    # Divide the dataset into different folds
    for i in range(0, k):
        # Divide the dataset
        if (i == 0):
            # First Fold
            XTrain = X[numberOfElements:numberOfTrainingExamples][:]
            print(XTrain.shape)
            yTrain = y[numberOfElements:numberOfTrainingExamples]
            XValid = X[0:numberOfElements][:]
            yValid = y[0:numberOfElements]
        elif (i == (k - 1)):
            # k-th Fold
            XTrain = X[0:numberOfTrainingExamples - numberOfElements][:]
            yTrain = y[0:numberOfTrainingExamples - numberOfElements]
            XValid = X[(k - 1) * (numberOfElements):numberOfTrainingExamples][:]
            yValid = y[(k - 1) * (numberOfElements):numberOfTrainingExamples]
        else:
            XTrain1 = X[0:(i) * (numberOfElements)][:]
            yTrain1 = y[0:(i) * (numberOfElements)]
            XTrain2 = X[(i + 1) * (numberOfElements):numberOfTrainingExamples][:]
            yTrain2 = y[(i + 1) * (numberOfElements):numberOfTrainingExamples]
            XTrain = np.concatenate((XTrain1, XTrain2), axis = 0)
            yTrain = np.concatenate((yTrain1, yTrain2), axis = 0)
            XValid = X[(i * numberOfElements):(i + 1) * numberOfElements][:]
            yValid = y[(i * numberOfElements):(i + 1) * numberOfElements]
       
        start_time=time.time()
        # k-Fold Cross Validation
        # Train for (k-1) datasets
        val_acc = train_svm(XTrain,yTrain,XValid,yValid,C)
        print(val_acc)
        all_acc.append(val_acc)
        end_time=time.time()
        
        #print ("accuracy={}={} ".format(val_acc))
        
    print("Final Accuracy at C= {}={}".format(C,sum(all_acc)/len(all_acc)))
    print('Time: ',end_time - start_time, 's')
    return sum(all_acc)/len(all_acc)


# In[ ]:


C=[(4**(-6)),(4**(-5)),(4**(-4)),(4**(-3)),(4**(-2)), (4**(-1)), 1, 4,(4**(2)), (4**(3)), (4**(4)), (4**(5)), (4**(6))]
for c in C:
    k_FoldCrossValidation(train_data1, train_label, k = 5,C=c)


# In[ ]:



C = 4**(-4)
p = fit(train_data1, train_label, C)
    
wP = np.array(p['x'][:train_data.shape[1]])
bP = p['x'][train_data.shape[1]]
def test_svm(test_data, test_label, w, b):
    xValidate = test_data.to_numpy()
    yP = xValidate.dot(wP) + bP
    a=yP
    a[a<0]=(-1)
    a[a>0]=1
    print(accuracy_score(np.round(a), test_label.to_numpy()))


# In[ ]:


test_svm(test_data1, test_label, wP, bP)

