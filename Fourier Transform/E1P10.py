import numpy as np
import matplotlib.pyplot as plt

def f(x):
	f_ = np.zeros_like(x)
	for i in range(len(x)):
		if (0 <= x[i] < np.pi):
			f_[i] = 1
	return f_

def fP(x,N):
	fP_ = np.zeros_like(x) + 1/2.0
	for i in range(len(x)):
		for n in range(1,N):
			fP_[i] += 1/(np.pi) * (1 - (-1)**n)/float(n) * np.sin(n*x[i])
	return fP_
	
	
	return fP_

n = 100	
x = np.linspace(0,2*np.pi,n+1)

plt.plot(x,f(x),'k')
plt.plot(x,fP(x,1),'y')
plt.plot(x,fP(x,2),'c')
plt.plot(x,fP(x,3),'g')
plt.plot(x,fP(x,10),'b')
plt.plot(x,fP(x,100),'r')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fourier transform of a Step function')
plt.legend(['exact','N = 1','N = 2','N = 3','N = 10','N = 100',])
plt.savefig('FT_of_step_function.png')
plt.show()
