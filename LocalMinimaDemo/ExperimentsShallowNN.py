'''
Created on Sep 4, 2015

@author: Iaroslav Shcherbatyi
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import random as random

Reps = 1 # number of times to repeat the experiment
M = 3 # number of dimensions
Xsz = 100; # number of training instances
Changes = [1,10,100,1000]

# compute single hidden layer neural network
def HLayer(W,X):
    return np.maximum( np.dot(X,W[0:M,]) , 0)
    #return np.tanh( np.dot(X,W[0:M,]) )
    
def NN(W,X):
    return np.dot(HLayer( W ,X), W[-1,] )

def L2NN(W, X, Y):
    return np.linalg.norm(NN(W,X) - Y, ord=2)  ** 2

def L2FixedNN(W, X, Y):
    W[-1,:] = np.linalg.lstsq(HLayer(W,X), Y)[0] # compute least squares
    return W, L2NN(W, X, Y) # compute objective

fvals = np.zeros((len(Changes), Xsz-1))
#fvalsRs = np.zeros(Xsz-1)
#fvalsGr = np.zeros(Xsz-1)
iters = np.zeros((len(Changes), Xsz-1))

impr = 0;
nimpr = 0;

for chidx in range(len(Changes)):

    for rep in range(Reps):
        
        fv = 10 ** 10
        X = np.random.randn(Xsz, M) # input features
        X[:,M-1] = 1
        Y = np.sin( X[:,0]*3 ) * np.sin( X[:,1]*3 ) # some nonlinear function
        # neuron weights with output weight in column format. Start with 1 neuron
        W = np.random.randn(M+1,1)
        G = HLayer(W,X)
        
        for N in range(Xsz-1): # number of neurons in the range 1 .. 99
            # add one neuron to network
            W = np.column_stack((W, W[:,0]*0))
            
            # sample random neuron values until fval is not improved
            improved = 1;
            #print "start"
            while (improved > 0):
                W[:,-1] = np.random.randn(M+1); # generate random neuron
                W, fval = L2FixedNN(W, X, Y) # compute fixed neurons objective
                if fval < fv:
                    improved-=1
                    Wb, fv = np.copy(W), fval
                    impr+=1
                else:
                    G = HLayer(W,X)
                    nimpr+=1
                    #print G[:,-1]"
            
            W = Wb
            
            #print "end"
            # do random permutations of neurons
        
            
            Wperm = np.copy(W)   
             
            for i in range(Changes[chidx]):
                change = np.random.randn(X.shape[1]+1);
                idx = random.randint(1,Wperm.shape[1]-1)
                Wperm[:,idx] += change # random permutation
                Wperm, fval = L2FixedNN(Wperm, X, Y) # compute fixed neurons objective
                if fval < fv:
                    fv = fval
                    W = np.copy(Wperm)
                else:
                    Wperm[:,idx] -= change
            
            
            # compute gradient descent
            # fvalsGr[N] = optim.minimize(lambda Wfl: L2NN(np.reshape(Wfl,  W.shape), X, Y) , W.flatten()).fun
            
            fvals[chidx,N] += fv
            iters[chidx,N] = N+1
            print N, fvals[chidx,N]

#print "improved:", impr, "not improved:", nimpr

# plot RESULTS
fig,ax = plt.subplots(figsize=(10,4))
#ax.plot(iters[0], np.log10( fvals[0] / Reps ),'-')

for chidx in range(len(Changes)):
    ax.plot(iters[chidx,:90], np.log10( fvals[chidx,:90] / Reps ),'-', label=( `Changes[chidx]` + " changes"))

ax.legend(loc='upper right', shadow=False)
#ax.plot(iters, np.log10( fvalsRs / Reps ),'-')
#ax.plot(iters, np.log10( fvalsGr / Reps ),'-')
ax.set_xlabel('Neurons')
ax.set_ylabel(' log(Objective) ')
fig.show()
plt.grid()
plt.show()