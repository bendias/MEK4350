import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" Getting the data from the files: """

BOB = 'BayOfBiscay.mat'
NYW = 'NewYearWave.mat'
file_BOB = h5py.File(BOB,'r') # Reading the data
file_NYW = h5py.File(NYW,'r') # Reading the data


t_BOB = file_BOB['t'][0]						# getting out the time array
eta_BOB = file_BOB['eta'][0]					# getting out the elevation array
nT_BOB = len(t_BOB)								# getting nT
dt_BOB = t_BOB[1] - t_BOB	[0]					# The length per step

t_NYW = file_NYW['t'][0]						# getting out the time array
eta_NYW = file_NYW['eta'][0]					# getting out the elevation array
nT_NYW = len(t_NYW)								# getting nT
dt_NYW = t_NYW[1] - t_NYW[0]					# The length per step

file_BOB.close()							# Closing the file BOB
file_NYW.close()							# Closing the file NYW


""" Plotting the elevation against time: """

#plt.plot(t_BOB,eta_BOB,'r')		# This will give the correct result
#plt.xlabel('time [s]')
#plt.ylabel('elevation [m]')
#plt.title('The elevation')
#plt.savefig('BayOfBiscay.png')
#plt.figure()
#plt.plot(t_NYW,eta_NYW,'r')		# This will give the correct result
#plt.xlabel('time [s]')
#plt.ylabel('elevation [m]')
#plt.title('The elevation')
#plt.savefig('NewYearWave.png')
#plt.show()

""" The significant wave heights: """

H_S_std_BoB = 4*np.sqrt(np.var(eta_BOB))
H_S_std_NYW = 4*np.sqrt(np.var(eta_NYW))
print ''
print 'Task b)'
print ''
print 'The significant waveheight, computed from std of the sampling data for BOB, is given by %g'%H_S_std_BoB
print 'The significant waveheight, computed from std of the sampling data for NYW, is given by %g'%H_S_std_NYW
print ''
""" df and NQ limit: """

df_BOB   = 1/(nT_BOB*dt_BOB)						# df is given by 1/T
NQ_T_BOB = 1/(2.0*dt_BOB)							# The Nyquist limit for BOB 

f_min_BOB =0 - 1/(2*dt_BOB)							# dt is already float
f_BOB = np.zeros(nT_BOB)							# Empty array to fil in
for i in range(0,nT_BOB):
	f_BOB[i] = i*df_BOB + f_min_BOB

df_NYW   = 1/(nT_NYW*dt_NYW)						# df is given by 1/T
NQ_T_NYW = 1/(2.0*dt_NYW)							# The Nyquist limit for NYW 

f_min_NYW =0 - 1/(2*dt_NYW)							# dt is already float
f_NYW = np.zeros(nT_NYW)							# Empty array to fil in
for i in range(0,nT_NYW):
	f_NYW[i] = i*df_NYW + f_min_NYW

print ''
print 'Task c)'
print ''
print 'df for BOB is given by df = %g with the following NyQuist limit: NQL = %g' %(df_BOB,NQ_T_BOB)
print 'df for NYW is given by df = %g with the following NyQuist limit: NQL = %g' %(df_NYW,NQ_T_NYW)
print ''

""" Estimation of the wave spectrum for positive frequency: """

norm_BOB = nT_BOB**2 * df_BOB													# Normalizing
F_BOB = abs(np.fft.fftshift(np.fft.fft(eta_BOB)))**2 / float(norm_BOB)			# The estimation of the spectrum for BOB
F_BOB = F_BOB[int(nT_BOB/2.0):]													# Only considering positive frequencies
f_pos_BOB = f_BOB[int(nT_BOB/2.0):]												# Only considering positive frequencies

norm_NYW = nT_NYW**2 * df_NYW													# Normalizing
F_NYW = abs(np.fft.fftshift(np.fft.fft(eta_NYW)))**2 / float(norm_NYW)			# The estimation of the spectrum for NYW
F_NYW = F_NYW[int(nT_NYW/2.0):]													# Only considering positive frequencies
f_pos_NYW = f_NYW[int(nT_NYW/2.0):]												# Only considering positive frequencies


#plt.plot(f_pos_BOB, F_BOB,'r')		# This will give the correct result
#plt.xlabel('f')
#plt.ylabel('S(f)')
#plt.title('Spectrum Of Bay Of Biscay')
#plt.savefig('SpectrumOfBayOfBiscay.png')
#plt.figure()						

#plt.plot(f_pos_NYW,F_NYW,'r')		# This will give the correct result
#plt.xlabel('f')
#plt.ylabel('S(f)')
#plt.title('Spectrum Of New Year Wave')
#plt.savefig('SpectrumOfNewYearWave.png')
#plt.show()						

""" Spectral Moments, Significant Wave Height and Mean Periods: """


def spectral_moment_BOB(j):
	m = 0
	f_ = f_BOB[int(len(f_BOB)/2.0):]
	
	for i in range(1,int(len(f_BOB)/2.0)):
		m += f_[i]**j * F_BOB[i] * df_BOB
	return 2*m


def spectral_moment_NYW(j):
	m = 0
	f_ = f_NYW[int(len(f_NYW)/2.0):]
	
	for i in range(1,int(len(f_NYW)/2.0)):
		m += f_[i]**j * F_NYW[i] * df_NYW
	return 2*m



m_0_check_BOB = np.var(eta_BOB)
m_0_BOB = spectral_moment_BOB(0)
m_1_BOB = spectral_moment_BOB(1)
m_2_BOB = spectral_moment_BOB(2)
m_n1_BOB= spectral_moment_BOB(-1)

m_0_check_NYW = np.var(eta_NYW)
m_0_NYW = spectral_moment_NYW(0)
m_1_NYW = spectral_moment_NYW(1)
m_2_NYW = spectral_moment_NYW(2)
m_n1_NYW= spectral_moment_NYW(-1)
print ''
print 'Task f)'
print ''
print 'std_BOB = %g and std_NYW = %g '%(m_0_check_BOB,m_0_check_NYW)
print 'm_0_BOB = %g and m_0_BOB = %g '%(m_0_BOB,m_0_NYW)
print 'm_1_BOB = %g and m_1_NYW = %g'%(m_1_BOB,m_1_NYW)
print 'm_2_BOB = %g and m_2_NYW = %g'%(m_2_BOB,m_2_NYW)
print 'm_n1_BOB= %g and m_n1_NYW= %g'%(m_n1_BOB,m_n1_NYW)

msg = 'We dont have corresponding values for m_0. It is a bug in spectral_moment method'
tol = 1e-10
diff = abs(m_0_BOB - m_0_check_BOB)
#assert diff < tol, msg

Hs_spm_BOB = 4*np.sqrt(m_0_BOB)
Hs_spm_NYW = 4*np.sqrt(m_0_NYW)

print ''
print 'Task g)'
print ''
print 'The significant waveheight for BOB is given by: %g '%Hs_spm_BOB
print 'The significant waveheight for NYW is given by: %g '%Hs_spm_NYW
print ''
print ''
print 'Task h)'
Tm01_BOB = m_0_BOB/m_1_BOB
Tm02_BOB = np.sqrt(m_1_BOB/m_2_BOB)
Tm01_NYW = m_0_NYW/m_1_NYW
Tm02_NYW = np.sqrt(m_1_NYW/m_2_NYW)
print ''
print 'The mean periods are given by:'
print 'Tm01_BOB = %g' % Tm01_BOB
print 'Tm02_BOB = %g' % Tm02_BOB
print 'Tm01_NYW = %g' % Tm01_NYW
print 'Tm02_NYW = %g' % Tm02_NYW
print ''
print ''
print ''






