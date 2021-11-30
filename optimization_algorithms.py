import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class SolveMinProb: #Main Minimization Class
 
    def __init__(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, mean, variance):
	
        self.matr_train = X_train
        self.matr_val = X_val
        self.matr_test = X_test
        self.vect_train = Y_train
        self.vect_val = Y_val
        self.vect_test = Y_test
        self.mean = mean
        self.variance = variance
        self.Np = self.vect_train.shape[0]
        self.Nf = self.matr_train.shape[1]
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        self.err = 0.0
        
    def MSE(self, dset='All'):
        self.MSE = np.zeros((3,1), dtype=float)
        (Y_train, Y_train_est) = ((self.variance * self.vect_train + self.mean) , (self.variance * np.dot(self.matr_train, self.sol) + self.mean))
        (Y_val, Y_val_est) = ((self.variance * self.vect_val + self.mean) , (self.variance * np.dot(self.matr_val, self.sol) + self.mean))
        (Y_test, Y_test_est) = ((self.variance*self.vect_test+self.mean) , (self.variance*np.dot(self.matr_test, self.sol)+self.mean))
        
        if dset=='All':
            MSE_train = (np.linalg.norm(Y_train - Y_train_est)**2)/self.matr_train.shape[0]
            MSE_val = (np.linalg.norm(Y_val - Y_val_est)**2)/self.matr_val.shape[0]
            MSE_test = (np.linalg.norm(Y_test - Y_test_est)**2)/self.matr_test.shape[0]
            self.MSE[0]=MSE_train
            self.MSE[1]=MSE_val
            self.MSE[2]=MSE_test
            return self.MSE[0], self.MSE[1], self.MSE[2] 
            
        elif dset=='val':
            MSE_train = (np.linalg.norm(Y_train - Y_train_est)**2)/self.matr_train.shape[0]
            MSE_val = (np.linalg.norm(Y_val - Y_val_est)**2)/self.matr_val.shape[0]
            self.MSE[0] = MSE_train
            self.MSE[1] = MSE_val
            return self.MSE[0], self.MSE[1]
        
        elif dset=='test':
            MSE_train = (np.linalg.norm(Y_train - Y_train_est)**2)/self.matr_train.shape[0]
            MSE_test = (np.linalg.norm(Y_test - Y_test_est)**2)/self.matr_test.shape[0]
            self.MSE[0] = MSE_train
            self.MSE[2] = MSE_test
            return self.MSE[0], self.MSE[2]
    
    def r2(self):       
        y_test = self.variance * self.vect_test + self.mean
        y_test_estimated = self.variance * np.dot(self.matr_test, self.sol) + self.mean

        
        r2=r2_score(y_test , y_test_estimated)  
        return r2
    
    def plot_err(self, title, logy, logx, xlim, ylim):

        error = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(error[:,0], error[:,1], label='Training', color='green')
            plt.plot(error[:,0], error[:,2], label='Validation', color='mediumblue')
        if (logy == 1) & (logx == 0):
            plt.semilogy(error[:,0], error[:,1], label='Training', color='green')
            plt.semilogy(error[:,0], error[:,2], label='Validation', color='mediumblue')           
        if (logy == 0) & (logx == 1):
            plt.semilogx(error[:,0], error[:,1], label='Training', color='green')
            plt.semilogx(error[:,0], error[:,2], label='Validation', color='mediumblue')          
        if (logy == 1) & (logx == 1):
            plt.loglog(error[:,0], error[:,1], label='Training', color='green')
            plt.loglog(error[:,0], error[:,2], label='Validation', color='mediumblue')
  
        plt.xlabel('n')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel('e(n)')
        plt.legend()
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.savefig(title + '.png', dpi=400)
        plt.show()

    def print_result(self, title):

        print(title,' : \n')
        print('The Optimum Weight Vector is: ')
        print(self.sol)
        print('The Obtained Minimum Squared Error is: ',self.min)
    def plot_weights(self, title, features):

        w_hat=self.sol         
        plt.figure(figsize=(10,6)) 
        plt.plot(w_hat)
        plt.xticks(np.arange(self.Nf), features, rotation=90)
        plt.xticks(np.arange(0, self.Nf, 1))
        plt.ylabel('W_Hat(n)')
        plt.grid() 
        plt.title(title)
        plt.savefig(title + '_W_Hat.png', dpi=400, bbox_inches='tight')
        plt.show()
    def weights(self):
        return self.sol

class LLS_Model(SolveMinProb):v #Least Linear Squares Algorithm
    
    def run(self):
        
        self.err = np.zeros((1,4), dtype=float)
        self.mse = np.zeros((1,4), dtype=float)
        
        A_train = self.matr_train
        y_train = self.vect_train
        
        A_val = self.matr_val
        y_val = self.vect_val
        
        A_test = self.matr_test
        y_test = self.vect_test
        
        weights = np.dot(np.dot(np.linalg.inv(np.dot(A_train.T, A_train)), A_train.T), y_train)
        
        self.mse[0,1] = np.linalg.norm(y_train - np.dot(A_train,weights))**2/A_train.shape[0]
        self.mse[0,2] = np.linalg.norm(y_val - np.dot(A_val,weights))**2/A_val.shape[0]
        self.mse[0,3] = np.linalg.norm(y_test - np.dot(A_test,weights))**2/A_test.shape[0]
        
        self.sol = weights
        self.err = self.mse
        
        return self.err[-1,1], self.err[-1,2], self.err[-1,3]

class Adam_SGD_Model(SolveMinProb): #Stochastic Gradient Descent with Adam Optimization
        
    def run(self, lr_rate=0.001, bet1=0.9, bet2=0.999, eps=1e-8, its=10000):
        
        
        self.err = np.zeros((its,4), dtype=float)
        self.mse = np.zeros((its,4), dtype=float)
        self.lr_rate = lr_rate
        self.bet1 = bet1
        self.bet2 = bet2
        self.eps = eps
        self.v = 0
        self.s = 0
        self.its = its
        weights = np.random.rand(self.Nf,1) # Random Initialization of the Weight Vector
        
        A_train = self.matr_train
        y_train = self.vect_train
        
        A_val = self.matr_val
        y_val = self.vect_val
        
        A_test = self.matr_test
        y_test = self.vect_test
        
        early_stopping = 0

        for it in range(its):
            if(early_stopping<30):
                gradient = 2 * np.dot(A_train.T, (np.dot(A_train, weights) - y_train))
            
                self.v = (self.bet1 * self.v) + (1 - self.bet1)*gradient # Momentum
                v_hat = self.v / (1 - (self.bet1**(it+1)))
                self.s = (self.bet2 * self.s) + (1 - self.bet2) * (gradient**2) #RMS Prop
                s_hat = self.s / (1 - (self.bet2**(it+1)))
                delta = self.lr_rate * (v_hat / (np.sqrt(s_hat) - self.eps)) # RMSProp + Momentum
                weights = weights - delta
            
                self.mse[it,0] = it
                self.mse[it,1] = np.linalg.norm(y_train - np.dot(A_train,weights))**2/A_train.shape[0]
                self.mse[it,2] = np.linalg.norm(y_val - np.dot(A_val,weights))**2/A_val.shape[0]
                self.mse[it,3] = np.linalg.norm(y_test - np.dot(A_test,weights))**2/A_test.shape[0]
                
                if (self.mse[it,2]>=self.mse[it-1,2]):
                    early_stopping+=1
                else:
                    early_stopping=0
            else:
                print('Number of Iterations for Adam SGD Model: ',it)
                break
        self.sol = weights
        self.err = self.mse
        
        return  self.err[-1,1], self.err[-1,2], self.err[-1,3]
        
        
class MGD_Model(SolveMinProb): #Mini Batch Gradient Descent Algorithm
    
    def run(self, lr_rate = 0.001, mb_size = 4, its = 10000):
        self.mse = np.zeros((its,4), dtype=float)
        self.mb_size = mb_size
        self.num_batches =  self.Np // self.mb_size
        self.err = np.zeros((its,2), dtype=float)
        self.lr_rate = lr_rate
        self.its = its
        
        A_train = self.matr_train
        y_train = self.vect_train
        
        A_val = self.matr_val
        y_val = self.vect_val
        
        A_test = self.matr_test
        y_test = self.vect_test
        
        
        weights = np.random.rand(self.Nf,1)# Random initialization of the weight vector
        
        early_stopping = 0

        for it in range(its):
            if (early_stopping<30):
                for batch_num in range(self.num_batches):
                
                    if self.Np % mb_size != 0 & batch_num == self.num_batches-1:
                        Am = A_train[batch_num*self.mb_size:]
                        ym = y_train[batch_num*self.mb_size]
            
                    else:
                        #seperating the features and labels miibatches in xm and ym
                        
                        Am = A_train[batch_num*self.mb_size:(batch_num+1)*self.mb_size]
                        ym = y_train[batch_num*self.mb_size:(batch_num+1)*self.mb_size]
                               
                    gradient = 2 * np.dot(Am.T, (np.dot(Am, weights) - ym))
                    weights = weights - lr_rate * gradient
                
                    self.mse[it,0] = it
                    self.mse[it,1] = np.linalg.norm(ym - np.dot(Am, weights))**2 / A_train.shape[0]
                    self.mse[it,2] = np.linalg.norm(y_val - np.dot(A_val, weights))**2 / A_val.shape[0]
                    self.mse[it,3] = np.linalg.norm(y_test - np.dot(A_test, weights))**2 / A_test.shape[0]

                if (self.mse[it,2]>=self.mse[it-1,2]):
                    early_stopping += 1
                else:
                    early_stopping = 0
            else:
                print('Number of Iterations for Mini Batch Gradient Decsent Model: ',it)
                break

        self.sol = weights
        self.err = self.mse
        
        return  self.err[-1,1], self.err[-1,2], self.err[-1,3]