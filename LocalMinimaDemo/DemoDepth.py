'''
Created on Sep 4, 2015

@author: Iaroslav Shcherbatyi
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optim
from numpy import disp

N = 9 # number of neurons
M = 4 # number of dimensions
Xsz = 1000; # number of training instances

# compute single hidden layer neural network
def NN(W,Wo,X):
    return np.dot(np.maximum( np.dot(X,W) , 0), Wo )

def L2NN(W, Wo, X, Y):
    return np.linalg.norm(NN(W,Wo,X) - Y, ord=2) ** 2

avg = 0;
iters = 0;
X = np.random.rand(Xsz, M+1)*2-1 # input features
X[:,2] = 1;
Y = np.sin( X[:,0]*3 ) * np.sin( X[:,1]*3 ) # some nonlinear function

Inp = np.copy(X)

M = Inp.shape[1]
W = np.random.randn(M,N) # neuron weights in column format
Wb = np.random.randn(N)

fv = L2NN(W, Wb, Inp, Y)

for depth in range(10000000):
    
    improved = False
    i = 0
    while i < 100:
        i = i + 1
        idx = randint(0, N-1)
        add = np.random.randn(M)
        
        W[:,idx] = W[:,idx] + add
        # solve the last layer to global optimality
        A = np.maximum( np.dot(Inp,W) , 0)
        Wo = np.linalg.lstsq(A, Y)[0]
        V = NN(W,Wo,Inp)
        
        fval = L2NN(W, Wo, Inp, Y)
        
        if fval < fv:
            fv = fval
            Wb = Wo
            improved = True
        else:
            W[:,idx] -= add    
    
    #
    if depth % 10 == 0:
        print "objective:", fv, "depth:", depth
    
    W0 = np.row_stack(( W, Wb ))
    W0 = np.reshape(W0, -1)
    
    def objective(Win):
        Wr = np.reshape(Win, (M+1, -1))
        return L2NN(Wr[ range(M), :], Wr[M,:], Inp, Y)

    val = objective(W0)
    #res = scipy.optimize.minimize(objective, W0);
    
    def callbck(data):
        print objective(data)
    
    solution = optim.minimize(objective, W0, callback=callbck, options={"maxiter":100} )
    
    if fv > solution.fun:
        Wr = solution.x
        Wr = np.reshape(solution.x, (M+1, -1))
        W = Wr[ range(M), :]
        Wb = Wr[ M, :]
        fv = solution.fun
    
    print "objective optimized:", fv
    
    #temp1 = L2NN(W, Wb, Inp, Y)
    V = np.maximum( np.dot(Inp,W) , 0)
    Inp = np.concatenate( (V, X), axis=1 ) 
    #Inp = np.copy(V)
    
    M = Inp.shape[1]
    W = np.zeros((Inp.shape[1],N)) # neuron weights in column format
    W[range(N),0] = Wb
    W[range(N),1] = -Wb
    Wb = np.zeros(N)
    Wb[0] = 1
    Wb[1] = -1
    #temp2 = L2NN(W, Wb, Inp, Y)
    #print temp1, temp2

"""
# everything
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y, c='r', marker='^')
ax.scatter(X[:,0], X[:,1], V, c='b', marker='*')
ax.set_xlabel('X_0')
ax.set_ylabel('X_1')
ax.set_zlabel('outputs')
plt.show()
"""