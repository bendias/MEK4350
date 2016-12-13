import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FichData = 'ReWFLin00000_00000.mat'
file_ = h5py.File(FichData,'r') # Reading the data

x = file_['x']		# Getting the x-values
y = file_['y']		# Getting the y-values
eta = file_['eta']


X = x[0,:]
Y = y[0,:]

Nx = len(X)
Ny = len(Y)

Eta2D = eta[:,:]
file_.close()

#plt.plot(Eta2D) 		# This will give a krussedull
plt.imshow(Eta2D)		# This will give the correct result
plt.savefig('Wind_spectrum.png')
plt.figure()				# To show

""" Need to construct the k: """

dx = X[1] - X[0]
dkx = 2*np.pi/(Nx*dx)

dy = Y[1] - Y[0]
dky = 2*np.pi/(Ny*dy)

k_x_min = -np.pi / dx
k_y_min = -np.pi / dy

kx = np.zeros(Nx-1)
ky = np.zeros(Ny-1)
for i in range(0,Nx-1):
	kx[i] = i*dkx + k_x_min
for j in range(0,Ny-1):
	ky[j] = j*dky + k_y_min

dkx = kx[1] - kx[0]
dky = ky[1] - ky[0]

# Now we have Nx,Ny, dkx and dky!


norm = (Nx*Ny)**2 * dkx*dky
F = abs(np.fft.fftshift(np.fft.fft2(Eta2D)))**2 / float(norm)

#plt.plot(F) 					# This will give a krussedull
plt.imshow(F)					# This will give the correct result
plt.axis([150,350,150,350]) 	# Zooming in
plt.savefig('Swell_spectrum.png')
plt.show()						# To show


varians1 = np.var(Eta2D)
print varians1
varians2 = sum(sum(F)*dkx*dky)
# or varians2 = np.sum(F)*dkx*dky
print varians2


Hs = 4*np.sqrt(varians1)

print Hs	
