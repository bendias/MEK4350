import numpy as np
import matplotlib.pyplot as plt


sigma = 1
n = 1000

def P_e(x):
	return 1 - (1 - np.exp(-x/float(2*sigma**2)))

def log_P_e(x):
	return -x/float(2*sigma**2)

x = np.linspace(0,10,n)

plt.plot(x,P_e(x),'r')
plt.xlabel('x-value')
plt.ylabel('P_e')
plt.title('P_e(x)')
plt.figure()
plt.plot(x,P_e(x),'r')
plt.loglog()
#plt.plot(x, log_P_e(x),'r')
plt.xlabel('x-value')
plt.ylabel('P_e')
plt.title('P_e(x), loglog')

plt.show()
