import numpy as np
from NeuralNet import *
import random

class NeuralClassifier:
   
    def __init__(self):
        pass

    def predict(self,X):
        num_test,dim=X.shape
        print 'num_test ',num_test
        W1,W2,b1,b2=self.model["W1"],self.model["W2"],self.model["b1"],self.model["b2"]
        Z1=W1.dot(X.T)+b1
        A1=np.zeros_like(Z1)
        A1=sigmoid(Z1)
        Z2=W2.dot(A1)+b2        
        return Z2.argmax(0)
        
            
    def train(self, X, y, learning_rate=1e-4, reg=1e-1, num_iters=100000, verbose=False):     
            
        num_train,dim=X.shape
        num_class=np.max(y)+1
        print "class ",num_class
        self.Xtr=X
        self.Ytr=y
        self.model=initialize_parameters_nn(dim,100,num_class)
        total_loss=0.0
        
        for j in range(4000):
          sample_size=num_train/5
          shuffled_index=range(num_train)
          random.shuffle(shuffled_index)
          for i in range(5):
                loss,grad=calculate_gradient_loss(self.Xtr[shuffled_index[sample_size*i:sample_size*(i+1)],:],self.model,self.Ytr[shuffled_index[sample_size*i:sample_size*(i+1)]],reg)
                self.model["W1"]-=learning_rate*grad["W1"]
                self.model["W2"]-=learning_rate*grad["W2"]
                self.model["b1"]-=learning_rate*grad["b1"]
                self.model["b2"]-=learning_rate*grad["b2"]
                #self.Xtr[j]-=learning_rate*grad["X"]
          if(j%100==0):
                 Y_pred=np.zeros(self.Ytr.shape[0],dtype=self.Ytr.dtype)
                 Y_pred=self.predict(self.Xtr)
                 print "\n\n\n**************************************************************************************\n"
                 print j,"  ",loss
                 print " prediction ",Y_pred[:100]
                 print " EFFICIENCY IN PREDICTION FOR ",np.mean(Y_pred==y)
                 print "\n**************************************************************************************\n"
            
