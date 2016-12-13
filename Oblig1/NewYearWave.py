import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):			# Reading the information from the given file 
	infile = open(filename,'r')		
	data = []				
	for line in infile:
		words = line.split()
		data.append(float(words[0]))
	infile.close()
	data = np.array(data)
	return data

def Interpolation(version = 'linear', plot = True):
	data = 26 - read_data('freak.data')		# Getting the amplitude where 26 is the mean level
	t = np.linspace(0,1200,len(data))		# time array for 20 minutes
	data_ = data[384:811]				# Zooming in the interesting part of the graph
	t_ = t[384:811]					# Zooming in the interesting part of the graph

	if (version == 'linear' and plot == True):
		plt.plot(t,data,'r')
		plt.title('Draupner wave records')
		plt.xlabel('Time (s)')
		plt.ylabel('Sureface elevation (m)')
		plt.grid(True)
		plt.savefig('Dreauper_wave_records.png')
		plt.figure()

		plt.plot(t_,data_,'b')		# dt = len(data)/1200 gives us that:
		plt.title('Draupner wave records')		# t[i] = 180 when i = 384		
		plt.xlabel('Time (s)')				# t[i] = 380 when i = 811
		plt.ylabel('Sureface elevation (m)')		# Therefore we plot t[384:811] against data[384:811]
		plt.grid(True)					
		plt.savefig('Dreauper_wave_records_zoomed.png')	
		plt.show()

	if (version == 'Fourier'):
		data_ft = np.fft.fft(data)			# Fourier transform of the sampling
		N = len(data)
		M = 1e7						# length of our new sampling, interpolated 
		interpolated_data = np.zeros(M) + 0j

		for n in range(N/2):
			interpolated_data[n] = data_ft[n]	# Filling the first half at the start of our expanded array
			interpolated_data[-n] = data_ft[-n]	# Filling the second half at the end  of our expanded array

		t_expanded = np.linspace(0,1200,M)					# New time array, with more steps
		data_interpolated = M/float(N) * np.fft.ifft(interpolated_data).real	# we're only interested in the real value,
		data_ =  data_interpolated[.21*1e7:.23*1e7]				# and we need to multiply with M/N to normalize
		t_ = t_expanded[.21*1e7:.23*1e7]	
					
		if plot:
			plt.plot(t_expanded,data_interpolated,'r')
			plt.title('Draupner wave records interpolated by D.F.T.')
			plt.xlabel('Time (s)')
			plt.ylabel('Sureface elevation (m)')
			plt.grid(True)
			plt.savefig('Dreauper_wave_records_interpolated.png')
			plt.figure()

			plt.plot(t_,data_,'b')						#dt = len(data)/1200 gives us that:
			plt.plot(t[550:575],data[550:575],'r*')	
			plt.title('Draupner wave records interpolated and zoomed in')	#t[i] = 180 when i = 384		
			plt.xlabel('Time (s)')						#t[i] = 380 when i = 811
			plt.ylabel('Sureface elevation (m)')				#Therefore we plot t[384:811] against data[384:811]
			plt.grid(True)					
			plt.legend(['Interpolated by Fourier','The actual measure points'],loc = 4)
			plt.savefig('Dreauper_wave_records_Fourier_zoomed.png')	
			plt.show()

def Fourier_coefficients( plot = True):

	def line(a,b,x):
		l_ = np.zeros_like(x)
		for i in range(len(x)):
			l_[i] = a*x[i]**b	
		return l_

	eta = 26 - read_data('freak.data')				
	N = len(eta)
	n = np.linspace(0,N-1,N)
	eta_tilda = abs(np.fft.fft(eta))/float(N)

	if plot:
		eta_tilda = eta_tilda[1e2:]
		n = n[1e2:]
		plt.plot(n,eta_tilda,'b')
		plt.plot(n,line(3.5e4,-2.5,n),'r')
		plt.loglog()
		plt.title('Fouriertransform of the sampling')
		plt.xlabel('n')
		plt.ylabel('Fourier coefficients')
		plt.grid(True)
		plt.savefig('Fourier_coefficients.png')
		plt.figure()

		plt.plot(n,eta_tilda**2,'b')
		plt.plot(n,line((3.5e4)**2,-2*2.5,n),'r')
		plt.loglog()
		plt.title('Wave spectrum')
		plt.xlabel('omega_n')
		plt.ylabel('Fourier coefficients squared')
		plt.grid(True)
		plt.savefig('Wave_spectrum.png')
		plt.show()


Interpolation('linear',True)
Interpolation('Fourier',True)
Fourier_coefficients(True)


