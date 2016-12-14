import numpy as np
import matplotlib.pyplot as plt

def l(x,a,b):
	eps = 1e-15
	return a*(x + eps)**b 
def f(x):
	return x

def g(x):
	return abs(x - np.pi)

def h(x):
	if isinstance(x,(float,int)):			# Checking if x is a scalar
		return np.cos(x)
	else:									# Otherwise we assume x is an array
		h_ = np.zeros_like(x)				# Ps. if callable(x): can be used to find out if x
		for i in range(len(x)):				# is a function. Used in the first project in INF5620 
			h_[i] = np.cos(x[i])
		return h_

N = 1024
L = 2*np.pi
dx = L/float(N)
x = np.linspace(0,L-dx,N)					# linspace includes the last term, but we want the 

f = f(x)
g = g(x)
h = h(x)

def plot_functions_directly():
	plt.plot(x,f,'b')
	plt.plot(x,g,'g')
	plt.plot(x,h,'r')
	plt.xlabel('x')
	plt.ylabel('f,g,h')
	plt.title('plotting of the first graph')
	plt.legend(['f(x) = x','g(x) = abs(x-pi)','h(x) = cos(x)'],loc = 2)
	plt.savefig('EIVP6_a.png')
	plt.show()

def plot_the_ft_of_the_functions():

	n = np.linspace(0,N-1,N)

	f_tilda = abs(np.fft.fft(f)) / float(len(n))
	g_tilda = abs(np.fft.fft(g)) / float(len(n))
	h_tilda = abs(np.fft.fft(h)) / float(len(n))
	
	n_ = n[:int(round(N/2.0))]
	b1 = 1; b2 = 2; b3 = 60

	plt.plot(n[:int(round(N/2.0))],np.fft.fftshift(f_tilda[:int(round(N/2.0))]),'b')
	plt.plot(n[:int(round(N/2.0))],np.fft.fftshift(g_tilda[:int(round(N/2.0))]),'g')
	plt.plot(n[:int(round(N/2.0))],np.fft.fftshift(h_tilda[:int(round(N/2.0))]),'r')
	plt.xlabel('n')
	plt.ylabel('Fourier_coefficients')
	plt.title('plotting the FT of the three graphs')
	plt.savefig('EIVP6_abs_of_FC.png')
	plt.figure()
	plt.plot(n_,(f_tilda[:int(round(N/2.0))]),'b')
	plt.plot(n_,(g_tilda[:int(round(N/2.0))]),'g')
	plt.plot(n_,(h_tilda[:int(round(N/2.0))]),'r')
	plt.plot(n_,l(n_,1,-b1),'k--')
	plt.plot(n_,l(n_,1,-b2),'k--')
	plt.plot(n_,l(n_,1,-b3),'k--')
	plt.xlabel('n')
	plt.ylabel('Fourier_coefficients')
	plt.title('plotting the FT of the three graphs in a loglog-plot')
	plt.loglog()
	plt.axis([1,1e3,1e-19,1e1])
	plt.legend(['f_tilda','g_tilda','h_tilda','f slope = %s'%b1,'g slope = %s'%b2,'h slope = %s'%b3],loc = 4)
	plt.savefig('EIVP6_abs_of_FC_in_loglog.png')
	plt.show()

#plot_functions_directly()
plot_the_ft_of_the_functions()
