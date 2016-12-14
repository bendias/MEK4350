import numpy as np
import matplotlib.pyplot as plt

n = 1000
def y(x,a,b):
	return a*x**b
x = np.linspace(0,1,n+1)
a = [3,1,3]
b = [3,0,1]
plt.plot(x,y(x,a[0],b[0]),'k')
plt.plot(x,y(x,a[1],b[1]),'r')
plt.plot(x,y(x,a[2],b[2]),'b')
plt.xlabel('x')
plt.ylabel('y(x) = a*x**b')
plt.title('Loglog plot of y(x) = a*x**b')
plt.loglog()
plt.legend(['a = %s and b = %s' %(a[0],b[0]),'a = %s and b = %s' %(a[1],b[1]),'a = %s and b = %s' %(a[2],b[2])],loc = 4)
plt.grid('on')
plt.savefig('linear_loglog_plot.png')
plt.show()

