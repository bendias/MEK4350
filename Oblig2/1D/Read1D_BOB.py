import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" Getting the data from the file: """
FichData = 'BayOfBiscay.mat'
file_ = h5py.File(FichData,'r') # Reading the data

t = file_['t'][0]						# getting out the time array
eta = file_['eta'][0]					# getting out the elevation array
nT = len(t)								# getting nT
dt = t[1] - t[0]						# The length per step
file_.close()							# Closing the file

""" Plotting the elevation against time: """
plt.plot(t,eta,'r')		# This will give the correct result
plt.xlabel('time [s]')
plt.ylabel('elevation [m]')
plt.title('The elevation')
plt.savefig('BayOfBiscay.png')
plt.show()													# To show

""" Finding the significant wave height from std of the elevation: """

Hs1 = 4 * np.sqrt(np.var(eta))
print 'The significant waveheight is given by: %g '%Hs1

""" Finding df and NQ limit: """

df = 1/(nT*dt)						# df is given by dw / 2pi, or directly
NQ_T = 1/(2.0*dt)					# The Nyquist limit in t-direction 

f_min =0 - 1/(2*dt)						# dt is already float
f = np.zeros(nT)						# Empty array to fil in
for i in range(0,nT):
	f[i] = i*df + f_min

""" Estimation the wave spectrum: """

norm = nT**2 * df
F = abs(np.fft.fftshift(np.fft.fft(eta)))**2 / float(norm)

#plt.plot(f[int(nT/2.0):],F[int(nT/2.0):],'r')		# This will give the correct result
#plt.xlabel('f')
#plt.ylabel('S(f)')
#plt.title('Spectrum Of Bay Of Biscay')
#plt.savefig('SpectrumOfBayOfBiscay.png')
#plt.show()						

""" Spectral moments, significant wave heights and mean periods: """
	
def spectral_moment(j):							# Kun positive frekvenser, sjekk f
	m = 0
	for i in range(int(len(F)/2.0),len(F)):
		if (f[i] != 0):
			m += f[i]**j * F[i] * df
	return m

m_0_check = np.var(eta)
m_0 = spectral_moment(0)
m_1 = spectral_moment(1)
m_2 = spectral_moment(2)
m_n1= spectral_moment(-1)

print 'm_0 = %g'%m_0
print 'm_0 = %g'%m_0_check
print 'm_1 = %g'%m_1
print 'm_2 = %g'%m_2
print 'm_n1= %g'%m_n1

msg = 'We dont have corresponding values for m_0. It is a bug in spectral_moment method'
tol = 1e-10
diff = abs(m_0 - m_0_check)
assert diff < tol, msg

Hs2 = 4*np.sqrt(m_0)

print 'The significant waveheight is given by: %g '%Hs2

Tm01 = m_0/m_1
Tm02 = np.sqrt(m_1/m_2)

print 'The mean period estimation is given by: Tm01 = %g '%Tm01
print 'The mean period estimation is given by: Tm02 = %g '%Tm02
