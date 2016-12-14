import numpy as np
import matplotlib.pyplot as plt

f1 = np.array([0,1,0,0])
f2 = np.array([1,0,0,0,0,0,0])
f3 = np.array([1,0,0,0,0,0,0,0])


f1_FFT = np.fft.fft(f1)
f1_IFFT = np.fft.ifft(f1_FFT)
f2_SHIFTED = np.fft.fftshift(f2)
f3_SHIFTED = np.fft.fftshift(f3)

def print_in_terminal():
	print ''
	print 'Let f1 = [0,1,0,0] such that: '
	print 'The fft of f1 is given by: %s' %f1_FFT
	print 'The ifft of f1_FFT is given by: %s' %f1_IFFT.real
	print 'And shifting f2 = %s and f3 = %s' %(f2,f3)
	print 'gives us:'
	print 'f2_SHIFTED = %s' %f2_SHIFTED
	print 'f3_SHIFTED = %s' %f3_SHIFTED
	print ''
	print 'Conclusion:'
	print ''
	print 'fft: B = 1 and the exponential is negative'
	print 'ifft: A = 1/N and the exponential is positive' 
	print 'r = N/2, rounded down to an integer'
	print ''

def write_in_file():

	filename = 'FFT_in_Numpy.txt'
	f_ = open(filename,'w')
	f_.write('Let f1 = [0,1,0,0] such that: \n \n')
	f_.write('The fft of f1 is given by: %s \n' %f1_FFT)
	f_.write('The ifft of f1_FFT is given by: %s \n' %f1_IFFT.real)
	f_.write('And shifting f2 = %s and f3 = %s \n' %(f2,f3))
	f_.write('gives us: \n')
	f_.write('f2_SHIFTED = %s \n' %f2_SHIFTED)
	f_.write('f3_SHIFTED = %s \n \n' %f3_SHIFTED)
	f_.write('Conclusion:\n \n')
	f_.write('fft: B = 1 and the exponential is negative')
	f_.write('ifft: A = 1/N and the exponential is positive')
	f_.write('r = N/2, rounded down to an integer')

	f_.close()
	
print_in_terminal()
write_in_file()
