import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" Reading information from the given file: """

FichData = 'Record3D_3.mat'
file_ = h5py.File(FichData,'r') # Reading the data

dt = float(file_['dt'][0]) 			# Delta t
dx = float(file_['dx'][0]) 			# Delta x
dy = float(file_['dy'][0]) 			# Delta y 
nt = int(file_['nt'][0]) 			# Timesteps  
nx = int(file_['nx'][0]) 			# Steps in x-direction 
ny = int(file_['ny'][0]) 			# Steps in y-direction
waves3d = file_['waves3d'] 			# Array/Matrix with wave-elevation


""" Checking the shape and plotting the waves at t=0: """

print 'The shape of the waves3d-matrix is:'
print np.shape(waves3d)

#plt.imshow(waves3d[0,:,:])						# plotting the result
#plt.savefig('Elevation_Records3D_1_t0.png')	# Saving it as a PNG-file
#plt.show()										# Show the result

""" Finding kx,ky and w by the given parameters: """


dkx = 2*np.pi/(nx*dx)				# Delta kx
dky = 2*np.pi/(ny*dy)				# Delta ky
dw  = 2*np.pi/(nt*dt)				# Delta w


kx_min = - np.pi / dx				# the minimum kx-value
ky_min = - np.pi / dy				# the minimum ky-value
w_min  = - np.pi / dt				# the minimum w-value

kx = np.zeros(nx - 1)				# Empty array to fill in kx-values
ky = np.zeros(ny - 1)				# Empty array to fill in ky-values
w  = np.zeros(nt + 1)				# Empty array to fill in w-values

for i in range(0,nx - 1):			# Filling in the kx-values
	kx[i] = i*dkx + kx_min
	
for i in range(0,ny - 1):			# Filling in the ky-values
	ky[i] = i*dky + ky_min
	
for i in range(0,nt + 1):			# Filling in the w-values
	w[i]  = i*dw + w_min

NQ_X = 1/(2.0*dx)					# The Nyquist limit in x-direction
NQ_Y = 1/(2.0*dy)					# The Nyquist limit in y-direction
NQ_T = 1/(2.0*dt)					# The Nyquist limit in t-direction 

print ''
print 'a)'
print ''
print 'dkx = %g and NQ_x = %g' %(dkx,NQ_X)
print 'dky = %g and NQ_y = %g' %(dky,NQ_Y)
print 'dw = %g and NQ_t = %g' %(dw,NQ_T)
print ''

""" Estimation the 3D spectrum 	F^(3)(kx,ky,w)"""

norm = (nx*ny*nt)**2 * dkx * dky * dw
F = abs(np.fft.fftshift(np.fft.fftn(waves3d[:,:,:])))**2 / float(norm)

""" Plotting the spectrum for w = constant >= 0.5 """


#plt.imshow(F[16,:,:])
#plt.xlabel('kx')
#plt.ylabel('ky')
#plt.title('spectrum for constant w')
#plt.savefig('spectrumW0.png')
#plt.figure()

#plt.imshow(F[32,:,:])
#plt.xlabel('kx')
#plt.ylabel('ky')
#plt.title('spectrum for constant w')
#plt.savefig('spectrumW1.png')
#plt.figure()

#plt.imshow(F[48,:,:])
#plt.xlabel('kx')
#plt.ylabel('ky')
#plt.title('spectrum for constant w')
#plt.savefig('spectrumW2.png')
#plt.show()

""" Plotting the spectrum for kx = 0 """

#plt.imshow(F[:,128,:])
#plt.xlabel('w')
#plt.ylabel('ky')
#plt.title('spectrum for kx = 0')
#plt.savefig('spectrumkx=0.png')
#plt.show()

""" Plotting the spectrum for ky = 0 """

#plt.imshow(F[:,:,128])					# Picking 128 since it's in the middle
#plt.xlabel('w')
#plt.ylabel('kx')
#plt.title('spectrum for ky = 0')
#plt.savefig('spectrumky=0.png')
#plt.show()

""" Computing the unambiguous wave number spectrum F^(2)_+(kx,ky): """

F_am = np.sum(F[int(nt/2.0):,:,:] * dw, axis = 0)


plt.imshow(F_am)					# Picking 128 since it's in the middle
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('unambiguous wave number spectrum')
plt.savefig('unam_w_n_s_3.png')
plt.show()

""" Significant wave heights: """

print ''
print 'h)'
print ''
print 'The significant waveheight is given by:'
# Method 1:

Variance1 = np.var(waves3d) 
Hs1 = 4*np.sqrt(Variance1)
print 'Method 1 gives: %g'%Hs1
# Method 2:

Variance2 = np.sum(F)*dkx*dky*dw
Hs2 = 4*np.sqrt(Variance2)
print 'Method 2 gives: %g'%Hs2

# Method 3:

def spectral_moment_0():
	m = 0
	for i in range(len(F_am[:,0])):
		for j in range(len(F_am[0,:])):
			m += F_am[i,j] * dkx*dky
	return m
m0 = spectral_moment_0()
Hs3 =  4*m0
print 'Method 3 gives: %g'%Hs3

file_.close()
