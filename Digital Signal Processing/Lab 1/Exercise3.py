import numpy as np
import scipy as sp
import librosa
import pywt
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import math
import random
import librosa.display
from scipy import signal

	#zerocrossing rate

	# creating function sgn in order to create the sum 
def signn(a, b):
    return np.abs(np.sign(a)-np.sign(b))

def Exercise_Execution(file_path, Frequency, windows, time):

	y, sr=librosa.core.load(file_path,sr = Frequency)
	# creating our input signal with its sampling frequency

	N = len(y)
	#print(N)   

	t = np.linspace(0, N/Frequency , N)
	plt.figure(1)
	plt.plot(t,y)
	plt.xlabel('t (x 10^(-4))')
	plt.ylabel('Signal')
	plt.title('Signal ' + str(file_path))
	plt.show()

	# creating the signal square (sqr) = (x[n])^2
	sqr = []
	sqr = y*y

	hamming_win = signal.hamming(400) 

	#we create hamming windows with length at 400
	for i in range ( 0 , Frequency , 400 ):
	    m = []
	    m = sqr[ i : (i+400)]
	    m = m*hamming_win
	    
	    z = np.convolve( sqr , m )

	'''
	#Fs = Frequency
	for i in range( 0, N-1):
	    n = i
	'''
	for j in range(len(windows)):
	    z = []
	    hamming_win = signal.hamming(windows[j])
	    energy = []

	    #sampling
	    for i in range(0, N, windows[j]):
	        m = []
	        m = sqr[i : (i + windows[j])]
	        z = np.convolve( m, hamming_win, mode='same')
	        energy = np.concatenate((energy, z), axis = None)
	    
	    t1 = np.linspace(0, (N + time[j])/Frequency , N + time[j])
	    
	    #these prints are helping to find the lengths of arrays 't1' and 'energy' in order to compare and make the right adjustments
	    #print('t1 '+str(len(t1)))
	    #print('energy '+str(len(energy)))

	    plt.figure(2)
	    plt.plot(t, y)
	    plt.plot(t1,energy)
	    plt.legend('Signal', 'Energy')
	    plt.show()



	exampl=[]
	for i in range(len(y)-1):
		exampl = np.concatenate((exampl, signn(y[i+1], y[i])), axis=None)

	for j in range(len(windows)):
	    z = []
	    rect_win = (1/(windows[j]))*signal.boxcar(windows[j])
	    energy = []

	    #sampling
	    for i in range(0, N, windows[j]):
	        m = []
	        m = exampl[i : (i + windows[j])]
	        z = np.convolve(rect_win, m , mode='same')
	        energy = np.concatenate((energy, z), axis = None)
	    
	    t1 = np.linspace(0, (N + time[j])/Frequency , N + time[j])
	    
	    plt.figure(3)
	    plt.plot(t, y)
	    plt.plot(t1,energy)
	    plt.legend('Signal', 'Zero Crossing Rate')
	    plt.show()
	    
#3.1

# arrays in order to plot the reults
# i put the in the 'main' section because for eace .wav file they change and it is more comfortable to put them here 
# it helps the programm to be more user friendly and abstract

windows = [40, 250, 400, 1000]

#'time' array in order to plot the results
time = [4, 234 ,284 ,484]	# in order for the plots (figures 2 and 3) to be displayed we had to adjust the 'time' array values in order to make the length of the arrays 't1' and 'energy' equal

#replace the path '/home/panoskpr/Documents/Signal_Processing_Labs/Lab1/speech_utterance.wav' with YOUR path the .wave file
Frequency = 16000
Exercise_Execution(r'speech_utterance.wav', Frequency, windows, time)


#3.2
#same as it we did in 3.1

windows = [40, 400, 800, 1102, 1323]
time = [0, 0, 400, 862, 0]
Frequency = 44100

Exercise_Execution(r'music.wav', Frequency, windows, time)