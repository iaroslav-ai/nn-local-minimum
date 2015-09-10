from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 100)
y = np.exp( -(x ** 2)  )
y2 = y + np.random.randn(100)*0.05

fig,ax = plt.subplots()
ax.plot(x, y,'-')
ax.plot(x, y2,'o')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
plt.show()