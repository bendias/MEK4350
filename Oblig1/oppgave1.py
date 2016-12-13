import numpy as np
import matplotlib.pyplot as plt


""" Parameters given in the exercise: """
n = 4						# number of mesh points
L = 2*np.pi					# length of our x-array
dx = L/float(n)					# one step
x = np.linspace(0,2*np.pi - dx,n)		# uniform distributed x-array, with the end point excluded
a = np.array([0,1,0,-1])			# the sampling points

def linear_interpolation(plot = True):		# A linear interpolation to show how we don't want our graph. It's not smooth enough
	if plot:
		plt.plot(x,a,'g')
		plt.title('Linear interpolation of samplingdata')
		plt.xlabel('x')
		plt.ylabel('f(x)')
		plt.axis([-.5,2*np.pi,-1.2,1.2])
		plt.savefig('Linear_Interpolation.png')
		plt.figure()


def fourier_interpolation(test = True, plot = True):

	aft = np.zeros(100) + 0j			# We fill 100 points make the interpolation in
	dx = L/100.0					# one step
	xx = np.linspace(0,L-dx,100)			# uniform distributed x-array, with the end point excluded
	a_ = np.fft.fft(a)				# fft of our sampling points

	aft[0] = a_[0]		# We want to fill the components of a as close to origo 
	aft[1] = a_[1]		# as it's possible. By this and the fact that Python has
	aft[-2] = a_[2]		# defined a cylcic permitution of this, we place the values
	aft[-1] = a_[3]		# at the first half, and the second half of the array

	b = 25*np.fft.ifft(aft)

	if test:		# A test to see if our imaginary part is under a very small number
	
		msg = 'Warning! We expect a very small imaginary value, which is NOT the case..'
		tol = 1e-15
		for i in range(n):
			assert (b[i].imag < tol), msg		
		
	if plot:
		plt.plot(xx,b.real,'g')
		plt.plot(x,a,'r*')
		plt.title('Interpolation by fourier of samplingdata')
		plt.xlabel('x')
		plt.ylabel('f(x)')
		plt.axis([-.5,2*np.pi,-1.2,1.2])
		plt.grid(True)
		plt.savefig('Interpolation_by_Fourier.png')
		plt.figure()


def differentiated_fourier_interpolated(test = True, plot = True):

	aft = np.zeros(100) + 0j			# We fill 100 points make the interpolation in
	dx = L/100.0					# one step
	xx = np.linspace(0,L-dx,100)			# uniform distributed x-array, with the end point excluded
	a_ = np.fft.fft(a)				# fft of our sampling points

	aft[0] = a_[0]		# We want to fill the components of a as close to origo 
	aft[1] = a_[1]		# as it's possible. By this and the fact that Python has
	aft[-2] = a_[2]		# defined a cylcic permitution of this, we place the values
	aft[-1] = a_[3]		# at the first half, and the second half of the array

	daft = np.zeros_like(aft)
	
	daft[0] = a_[0] * 1j * 0
	daft[1] = a_[1] * 1j * 1
	daft[-2] = a_[2] * 1j * -2
	daft[-1] = a_[3] * 1j * -1

	b = 25*np.fft.ifft(daft)		# Normalize our sampling, remember that numpy divides on N
	if test:				# A test to see if our imaginary part is under a very small number
	
		msg = 'Warning! We expect a very small imaginary value, which is NOT the case..'
		tol = 1e-14
		for i in range(n):
			assert (b[i].imag < tol), msg		
		
	if plot:
		plt.plot(xx,b.real,'g')
		plt.plot(xx,np.cos(xx),'r--')
		plt.title('Interpolation by fourier of samplingdata')
		plt.xlabel('x')
		plt.ylabel('f(x)')
		plt.axis([-.5,2*np.pi,-1.2,1.2])
		plt.grid(True)
		plt.savefig('differatiated_sampling_Interpolation_by_Fourier.png')
	


linear_interpolation(plot = True)
fourier_interpolation(test = True, plot = True)
differentiated_fourier_interpolated(test = True,plot = True)

plt.show()


