import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

L = 10

delta = 0.025
x = np.arange(-L, L, delta)
y = np.arange(-L, L, delta)
X, Y = np.meshgrid(x, y)

def Z(x,y):
	return -y**2 - x**2 + 0.06 * x**3 

CS = plt.contour(X, Y, Z(X,Y))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour plot of psi = - y^2 + .06 x^3 - x^2 ')
plt.savefig('Contourplot.png')
plt.show()
