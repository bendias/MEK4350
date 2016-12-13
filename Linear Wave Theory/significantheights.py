import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
n = 1e3
N = np.linspace(1,n,n)
H_1N = lambda N:np.sqrt(8*np.log(N) + np.sqrt(2*np.pi)*N*sci.erf(np.sqrt(np.log(N))))
#def H_N(N):
#	return (np.sqrt(8*np.log(N) + np.sqrt(2*np.pi)*N*sci.erf(np.sqrt(np.log(N)))))

plt.plot(N,H_1N(N),'r')
plt.title('H_1/N')
plt.xlabel('N')
plt.ylabel('H_1/N')
plt.show()

