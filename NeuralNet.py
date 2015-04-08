import numpy as np
from data_utils import *
import random


def initialize_parameters_nn(input_size,hidden_size,output_size):


    model={}
    
    # W1 is  parameter weight  for the first layer of neural network including bias term as the parameter. shape (hidden_size,input_size+1)
    model["W1"]=np.random.randn(hidden_size,input_size)*.001
    """ W2 is  parameter weight  for the second layer of neural network including bias term as the parameter. shape           (output_size,hidden_size  +1)"""
    model["b1"]=np.zeros((hidden_size,1))
    model["W2"]=np.random.randn(output_size,hidden_size)*.001
    model["b2"]=np.zeros((output_size,1))
    return model
    
    
    
def calculate_gradient_loss(X,model,y=None,reg=.10):
    
    #unpacked the parameters from nn model dictionary
    W1,W2,b1,b2=model["W1"],model["W2"],model["b1"],model["b2"]
    num_sample=X.shape[0]
    #X=np.array([np.concatenate((np.array([1]),X))]).T
   
    loss=0.0
    Z1=W1.dot(X.T)+b1
    A1=np.zeros_like(Z1)
    A1=sigmoid(Z1)
    
    Z2=W2.dot(A1)+b2
    margin=Z2-Z2[y,range(num_sample)]+1
    margin[y,range(num_sample)]=0
    margin[margin<0]=0
    loss=np.sum(margin)/num_sample
    grad={}
    
    dZ2=np.zeros_like(Z2)
    
    dZ2[margin>0]=1
    dZ2[margin<0]=0
    dZ2[y,range(num_sample)]-=np.sum(margin>0,0)
    dZ2/=num_sample
    dW2=dZ2.dot(A1.T)
    db2=np.sum(dZ2,axis=1,keepdims=True)
    dA1=np.dot(W2.T,dZ2)
    #removing bias activation  
    dZ1=dsigmoid(Z1)*dA1
    dW1=dZ1.dot(X)
    db1=np.sum(dZ1,axis=1,keepdims=True)
    
    #ADDING REGULARIZATION TO WEIGHTS
    #dW1[:,1:]+=reg*W1[:,1:]
    #dW2[:,1:]+=reg*W2[:,1:]   
    
    
    grad["W1"]=dW1
    grad["W2"]=dW2
    grad["b1"]=db1
    grad["b2"]=db2
    
    return loss,grad
    

def sigmoid(X):
    return 1/(1+np.exp(-X*1.000))
    

def dsigmoid(X):
     return sigmoid(X)*(1-sigmoid(X))    
    
