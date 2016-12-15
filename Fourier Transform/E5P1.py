import numpy as np
import matplotlib.pyplot as plt


f = lambda x: x
g = lambda x: np.cos(x)

N = 1024
L = 2*np.pi
dx = L/N
x = np.linspace(0,L - dx,N)

def FT_by_brute_force():

	def f_j(x,M):
		f_ = np.zeros_like(x) + np.pi	
		for i in range(len(x)):
			for n in range(1,M+1):
				f_[i] += 2j/float(n) * np.exp(1j*n*x[i])
		return f_
	f_ = f_j(x,10)
	plt.plot(x,f(x),'k')
	plt.plot(x,f_j(x,1),'c')
	plt.plot(x,f_j(x,2),'y')
	plt.plot(x,f_j(x,3),'b')
	plt.plot(x,f_j(x,10),'g')
	plt.plot(x,f_j(x,100),'r')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.title('Fourier transform of a linear function')
	plt.legend(['exact','N = 1','N = 2','N = 3','N = 10','N = 100',],loc=4)
	plt.savefig('EVP1_by_brute_force.png')
	plt.show()

def FT_by_fft():
	f_ = f(x)
	f_1= np.zeros_like(x)
	f_tilda = np.fft.fft(f_)/float(N)
	for i in range(len(x)):
		for n in range(len(f_tilda)):
			f_1[i] += f_tilda[n] * np.exp(1j*n*x[i])
	plt.plot(x,f_1,'r')
	plt.plot(x,f_2.imag,'k')
	plt.show()
	
	
FT_by_brute_force()
#FT_by_fft()
