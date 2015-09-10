import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 200)
y = np.exp( -(x ** 2)/2  ) - np.exp( -(x ** 2)  )*2 + np.exp( -(x ** 2)*4  ) *2.5- np.exp( -(x ** 2)*8  )*1.6

# multiplier / offset / output weights
# bad initialization
W = np.array([ [1,1,1,1], [-2,-1,1,2], [1,1,1,1] ])
# initialization that leads to global optimum
#W = np.array([ [0.7,1,2,5], [0,0,0,0], [1,-2,2.5,-1.6] ])
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

yp = ComputeGaussians(W, x);

fig,ax = plt.subplots()
# plot data and predictions
ax.plot(x, y,'*')
ax.plot(x, yp,'-')
# plot Gaussian positions
xp = W[1,] / W[0,] # location of Gaussian

ax.plot(xp, xp*0,'o')

# plot gaussians, if necessary

"""
for i in range(4):
    ax.plot(x, ComputeGaussians(W[:,[i]], x), '-') """

ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
plt.grid()
plt.show()