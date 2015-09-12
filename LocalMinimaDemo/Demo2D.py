"""
This code computes the gradient descent for Gaussian mixture model and plots
result of fitting 
"""

import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 200)
y = np.exp( -(x ** 2)/2  ) - np.exp( -(x ** 2)  )*2 + np.exp( -(x ** 2)/0.25  ) *2.5- np.exp( -(x ** 2)/0.125  )*1.6

# deviation / mean / output weights
#W = np.array([ [0.2], [1.2], [1] ])# best model by enumeration (by hand :D )
W = np.array([ [1,0.3,0.3,1], [-2,-0.3,0.3,2], [1,1,1,1] ])# bad initialization
#W = np.array([ [1.7,1,0.5,0.35], [0,0,0,0], [1,-2,2.5,-1.6] ])# initialization that leads to global optimum
#W = np.array([ [1,0.3,0.3,1, 0.3, 0.3, 0.2], [-2,-0.3,0.3,2, -0.7, 0.7, 0], [1,1,1,1,1,1,1] ])# better results with more neurons

# Here I translate the parameters to more useful form
tW = np.abs(W[0,])
W[0,] = 1 / tW
W[1,] = W[1,] / tW

shapeOfW = W.shape
W0 = np.squeeze(np.asarray(W))

# this is required for optimization procedure
def Unflatten(Win):
    return np.reshape(Win, shapeOfW)

def ComputeGaussians(W, X):
    return np.dot( np.exp( -(np.outer(X, W[0,:]) - W[1,]) ** 2 ) , W[2,])

def Objective(Win):
    return np.linalg.norm( ComputeGaussians(Unflatten(Win), x) - y) ** 2 # Sq. L2 norm

sol = opt.minimize(Objective, W0)
W = Unflatten(sol.x)
print "solution objective:", Objective(W)

yp = ComputeGaussians(W, x);

fig,ax = plt.subplots(figsize=(10,4))
# plot data and predictions
ax.plot(x, y,'*')
ax.plot(x, yp,'-')
# plot Gaussian positions
xp = W[1,] / W[0,] # location of Gaussian

ax.plot(xp, xp*0,'o')

# plot separate gaussians, if necessary

"""
for i in range(4):
    ax.plot(x, ComputeGaussians(W[:,[i]], x), '-') """

ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
plt.grid()
plt.show()