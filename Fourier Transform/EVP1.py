import numpy as np
import matplotlib.pyplot as plt


f = lambda x: x
g = lambda x: np.cos(x)

N = 1024
L = 2*np.pi
dx = L/N
x = np.linspace(0,L - dx,N)

def f_j(x,M):
	f_ = np.zeros_like(x) + np.pi	
	for i in range(len(x)):
		for n in range(1,M+1):
			f_[i] += 2j/float(n) * np.exp(1j*n*x[i])
	return f_
"""
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
"""

def f_diff(x,M,alpha):
	f_ = np.zeros_like(x) + np.pi	
	for i in range(len(x)):
		for n in range(1,M+1):
			f_[i] +=(1j*n)**alpha* 2j/float(n) * np.exp(1j*n*x[i])
	return f_
	

plt.plot(x,f_diff(x,100,0),'k')
plt.plot(x,f_diff(x,100,0.5),'b')
plt.plot(x,f_diff(x,100,1),'r')
plt.plot(x,f_diff(x,100,1.5),'g')
plt.xlabel('x')
plt.ylabel('P_Nf(x)')
plt.title('f(x) fractional differentiated')
plt.legend(['alpha = 0','alpha = 1','alpha = .5','alpha = 1.5'],loc = 8)
plt.axis([0,2*np.pi,-300,300])
plt.savefig('EVP1_f_differentiated.png')
plt.show()
