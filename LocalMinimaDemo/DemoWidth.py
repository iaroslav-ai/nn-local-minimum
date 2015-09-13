'''
Created on Sep 4, 2015

@author: Iaroslav Shcherbatyi
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D

N = 10 # number of neurons
M = 2 # number of dimensions
Xsz = 1000; # number of training instances

# compute single hidden layer neural network
def NN(W,Wo,X):
    return np.dot(np.maximum( np.dot(X,W) , 0), Wo )

def L2NN(W, Wo, X, Y):
    return np.linalg.norm(NN(W,Wo,X) - Y, ord=2)  ** 2

avg = 0;
iters = 0;
Wb = np.random.rand(N)

for idx in range(1):
    X = np.random.rand(Xsz, M+1)*2-1 # input features
    X[:,2] = 1;
    Y = np.sin( X[:,0]*3 ) * np.sin( X[:,1]*3 ) # some nonlinear function
    
    W = np.random.rand(M+1,N)*2-1 # neuron weights in column format
    Wo = np.random.rand(N) # weights of output layer
    
    fv = 10 ** 10
    
    for i in range(10):
        
        idx = randint(0, N-1)
        add = np.random.rand(M+1)
        
        W[:,idx] = W[:,idx] + add
        # solve the last layer to global optimality
        A = np.maximum( np.dot(X,W) , 0)
        Wo = np.linalg.lstsq(A, Y)[0]
        V = NN(W,Wo,X)
        
        fval = L2NN(W, Wo, X, Y)
        
        if fval < fv:
            fv = fval
            Wb = Wo
        else:
            W[:,idx] -= add    
    
    
    print "objective:", fv
    avg = avg + fv;
    iters = iters + 1
    print "avg: ", avg / iters


# everything
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y, c='r', marker='^')
ax.scatter(X[:,0], X[:,1], NN(W,Wb,X), c='b', marker='*')
ax.set_xlabel('X_0')
ax.set_ylabel('X_1')
ax.set_zlabel('outputs')
plt.show()
