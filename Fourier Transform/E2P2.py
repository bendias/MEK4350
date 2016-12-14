import numpy as np
import matplotlib.pyplot as plt
n = 100000



def I(k,a):
	I_ = np.zeros_like(k)
	for i in range(len(k)):
		if (k[i] == 0 or a == 0):
			I_[i] = 1
		else:
			I_[i] = 2*a*np.sin(k[i]*a)/float(k[i]*a)

	return I_
	
k = np.linspace(-1,1,n+1)

plt.plot(k,I(k,.1),'r')
plt.plot(k,I(k,10),'b')
plt.plot(k,I(k,100),'k')
plt.xlabel('k')
plt.ylabel('I(k,a)')
plt.title('Drawing of 2*a*sinc(k*a)')
plt.legend(['a = small','a = moderate','a = large'])
plt.savefig('Drawing_of_sinc.png')
plt.show()
