# %matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal

X = np.linspace(-5, 5, 50)

Y = np.linspace(-5, 5, 50)

X, Y = np.meshgrid(X, Y)

X_mean = 0
Y_mean = 0

X_var = 5
Y_var = 8

pos = np.empty(X.shape + (2, ))

pos[:, :, 0] = X

pos[:, :, 1] = Y

rv = multivariate_normal([X_mean, Y_mean], [[X_var, 0], [0, Y_var]])

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, rv.pdf(pos), cmap="plasma")

plt.show()
