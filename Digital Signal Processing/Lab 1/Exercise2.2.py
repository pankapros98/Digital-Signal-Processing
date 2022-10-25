import numpy as np
import scipy as sp
import librosa
import pywt
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import math
from scipy import signal

#2.2.a

n = np.linspace(0,2000,2000)
#linspace() returns evenly spaced numbers,(start, stop , number of samples to generate)

t = n/1000
w2 = 1.5*np.cos(2*math.pi*80*t) + 1.7*signal.unit_impulse(2000, 725) + 1.7*signal.unit_impulse(2000, 900) + 0.15*np.random.normal(0,1,2000)
#normal() Draws random samples from a normal (Gaussian) distribution, (centre,width,size)

#%matplotlib inline
plt.figure(1)
plt.plot( n, w2 )
plt.xlim([300, 950])# , χρησιμοποιούμε αυτή την εντολή για να γίνουν πιο ευδιάκριτες οι 
# απότομες μεταβολές του σηματος στις χρονικές στιγμές  725msec και 900msec

plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('signal 2.2.a')
plt.show()
#2.2.b

#D = np.abs(librosa.stft(w2, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w2, 2048, 20, 40)
print(D.shape)
print(w2)
plt.figure(2)
plt.plot(D)
plt.show()
#2.2.b

#%matplotlib inline
t = np.linspace(0,100000/22050, 101)
f = np.linspace(0,11025,1025)
plt.figure(3)
plt.contour( t,f, np.abs(D), 15 )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of w2')
plt.xlim([ 0 , 4 ])
plt.show()
#2.2.c1

#D = np.abs(librosa.stft(w2, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w2, 2048, 40, 80)
print(D.shape)
print(w2)
plt.figure(4)
plt.plot(D)
plt.show()
#2.2.c1

#%matplotlib inline
t = np.linspace(0,100000/22050, 51)
f = np.linspace(0,11025,1025)
plt.figure(5)
plt.contour( t,f, np.abs(D), 15 )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of w2')
plt.xlim([ 0 , 4 ])
plt.show()
#2.2.c2

#D = np.abs(librosa.stft(w2, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w2, 2048, 80, 160)
print(D.shape)
print(w2)
plt.figure(6)
plt.plot(D)
plt.show()
#2.2.c2

#%matplotlib inline
t = np.linspace(0,100000/22050, 26)
f = np.linspace(0,11025,1025)
plt.figure(7)
plt.contour( t,f, np.abs(D), 15 )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of w2')
plt.xlim([ 0 , 4 ])
plt.show()
#2.2.d

s = np.power(2,np.linspace(1,7,96))
coefs,freqs = pywt.cwt( w2 , s , 'cmor3.0-1.0')
#μορφη: pywt.cwt( data = input signal , scales , wavelet ) 

print(coefs.shape)
#2.2.d

#%matplotlib inline
t = np.linspace(0/1000,2000/1000,2000)
f = freqs*1000
plt.figure(8)
plt.contour( t,f, np.abs(coefs), 15 )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('CWT of w2')
plt.show()

plt.figure(9)
plt.contour( t,s, np.abs(coefs), 13 )
plt.xlabel('Time (sec)')
plt.ylabel('s')
plt.title('CWT of w2')
plt.show()

