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
maxdepth = 1000

# compute single hidden layer neural network
def HLayer(W,X):
    result = np.dot( X , W[0:W.shape[0]-1, : ] )
    result[:,1:] = np.maximum(result[:,1:], 0)
    #result[1:,] = np.tanh(result[1:,], 0)
    return result
    
def NN(W,X):
    return np.dot(HLayer( W ,X), W[-1,] )

def L2NN(W, X, Y):
    return np.linalg.norm(NN(W,X) - Y, ord=2)  ** 2

def L2FixedNN(W, X, Y):
    W[-1,:] = np.linalg.lstsq(HLayer(W,X), Y)[0] # compute least squares
    fval = L2NN(W, X, Y)
    return W,  fval# compute objective

fvals = np.zeros(maxdepth)
fvalsRs = np.zeros(maxdepth)
fvalsGr = np.zeros(maxdepth)
iters = np.zeros(maxdepth)

for rep in range(Reps):
    
    fv = 10 ** 10
    X = np.random.randn(Xsz, M) # input features
    X[:,M-1] = 1
    Y = np.sin( X[:,0]*3 ) * np.sin( X[:,1]*3 ) # some nonlinear function
    # neuron weights with output weight in column format. Start with 1 neuron
    
    W = np.random.randn(X.shape[1]+1,1)
    Input = np.copy(X);
    
    for depth in range(maxdepth):
        
                
        for N in range(2):
            # add one neuron to network
            W = np.column_stack((W, W[:,0]*0))
            
            # sample random neuron values until fval is not improved
            improved = 1;
            #print "start"
            while (improved > 0):
                W[:,-1] = np.random.randn(Input.shape[1]+1); # generate random neuron
                W, fval = L2FixedNN(W, Input, Y) # compute fixed neurons objective
                if fval <= fv:
                    improved-=1
                    Wb, fv = np.copy(W), fval
            W = Wb
            
            #print "end"
            # do random permutations of neurons
        
            
            for i in range(10):
                change = np.random.randn(Input.shape[1]+1);
                idx = random.randint(1,W.shape[1]-1)
                W[:,idx] += change # random permutation
                W, fval = L2FixedNN(W, Input, Y) # compute fixed neurons objective
                if fval < fv:
                    fv = fval
                else:
                    W[:,idx] -= change
            
            # compute gradient descent
            # fvalsGr[N] = optim.minimize(lambda Wfl: L2NN(np.reshape(Wfl,  W.shape), Input, Y) , W.flatten()).fun
            
        
        print depth, fv
            
        fvals[depth] += fv
        iters[depth] = depth;
        W, fval = L2FixedNN(W, Input, Y)
        # replace X with outputs of neurons
        
        G = HLayer(W,Input);
        #s = W[-1,]
        #Input = np.concatenate((G, X[:, [-1]]), axis=1)
        
        s = np.concatenate( ( W[-1,], X[0,]*0 ) )
        Input = np.concatenate((G, X), axis=1)
        
        W = np.random.randn(Input.shape[1]+1,1)
        W[:s.shape[0],0] = s;
        W[-1,0] = 1;
        W, fval2 = L2FixedNN(W, Input, Y)


# plot RESULTS
fig,ax = plt.subplots(figsize=(10,4))
ax.plot(iters, ( fvals / Reps ),'-')
#ax.plot(iters, np.log10( fvalsRs / Reps ),'-')
#ax.plot(iters, np.log10( fvalsGr / Reps ),'-')
ax.set_xlabel('Layers')
ax.set_ylabel('log( Objective )')
fig.show()
plt.grid()
plt.show()