from data_utils import load_CIFAR10
from nn_classifier import *
import numpy as np
from math import pow

    
Xtr,Ytr,Xte,Yte=load_CIFAR10('dataset/')#loaded Cifar10 data set as training set Xtr, labels of training set as Ytr, Xte of training set,Yte of Training set 

Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

Xtr_rows=Xtr_rows-np.mean(Xtr_rows,0)
Xtr_rows=Xtr_rows/np.std(Xtr_rows,0)
for i in range(-2,-1,1):     
  nn=NeuralClassifier()
  nn.train(Xtr_rows,Ytr,pow(10,i),.001)
  Y_pred=np.zeros(Yte.shape[0],dtype=Ytr.dtype)
  Y_pred=nn.predict(Xte_rows)
  print "\n\n\n**************************************************************************************\n"
  print "prediction ",Y_pred[:100]
  print " EFFICIENCY IN PREDICTION FOR ",np.mean(Y_pred==Yte)
  print "\n**************************************************************************************\n"
