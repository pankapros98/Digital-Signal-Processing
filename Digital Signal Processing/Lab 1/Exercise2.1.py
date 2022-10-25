import numpy as np
import scipy as sp
import librosa
import pywt
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import math
# 2.1.a

n = np.linspace(0,2000,2000)
#n:(μεταβλητή) πλήθος δειγμάτων, λόγω συχνότητας δειγματοληψίας fs=1000Hz
#linspace() returns evenly spaced numbers,(start, stop , number of samples to generate)

t = n/1000
# t: μεταβλητή χρόνου 
w1 = 2*np.cos(2*math.pi*70*t) + 3*np.sin(2*math.pi*100*t) + 0.1*np.random.normal(0,1,2000)
#normal() Draw random samples from a normal (Gaussian) distribution, (centre,width,size)
# με την normal() παριστάνεται ο λευκός Gaussian θόρυβος μηδενικής μέσης τιμής v(t) 

#%matplotlib inline
plt.figure(1)
plt.plot( n, w1 )
plt.xlim([ 0, 2000 ])
# με την xlim αλλάζουμε το εύρος του άξονα x

plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('signal 2.1.a')
plt.show()
#2.1.b

#D = np.abs(librosa.stft(w1, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w1, 2048, 20, 40)
# D : STFT του σήματος w1
print(D.shape)
# με την εντολή shape παίρνουμε το πλήθος των δειγμάτων του stft για συχνότητα(1025) και χρονο(101) αντίστοιχα 
print(w1)
plt.figure(2)
plt.plot(D)
plt.show()
#2.1.b

#%matplotlib inline
t = np.linspace(0,100000/22050, 101)
f = np.linspace(0,11025,1025)
plt.figure(3)
plt.pcolormesh(t,f,np.abs(D))
# Με την plt.pcolormesh Δημιουργούμε μια γραφική παράσταση ψευδοχρώματος ( συχνότητας , χρόνου )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of signal w1')
plt.show()

#2.1.c1

#D = np.abs(librosa.stft(w1, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w1, 2048, 40, 80)
# D : STFT του σήματος w1
print(D.shape)
print(w1)
plt.figure(4)
plt.plot(D)
plt.show()

#2.1.c1

#%matplotlib inline
t = np.linspace(0,100000/22050, 51)
f = np.linspace(0,11025,1025)
plt.figure(5)
plt.pcolormesh(t,f,np.abs(D))
# Με την plt.pcolormesh Δημιουργούμε μια γραφική παράσταση ψευδοχρώματος ( συχνότητας , χρόνου )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of signal w1')
plt.show()
#2.1.c2

#D = np.abs(librosa.stft(w1, n_fft=2048, hop_length=20, win_length=40))

D = librosa.core.stft( w1, 2048, 80, 160)
# D : STFT του σήματος w1
print(D.shape)
print(w1)
plt.figure(6)
plt.plot(D)
plt.show()

#2.1.c2

#%matplotlib inline
t = np.linspace(0,100000/22050, 26)
f = np.linspace(0,11025,1025)
plt.figure(7)
plt.pcolormesh(t,f,np.abs(D))
# Με την plt.pcolormesh Δημιουργούμε μια γραφική παράσταση ψευδοχρώματος ( συχνότητας , χρόνου )

plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of signal w1')
plt.show()

#2.1.d

s = np.power(2,np.linspace(1,7,96))
#Κλίμακα s:βρίσκουμε τις οκτάβες για την συχνότητα δειγματοληψίας fs=1000Hz[fs(min)=15,625Hz],ως δυνάμεις του 2(με την np.power)
#Συνολικά , για την συχνότητα δειγματοληψίας fs=1000Hz,βρίσκουμε 6 οκτάβες [ απο την σχεση fs/(2^x) = fmin]
coefs,freqs = pywt.cwt( w1 , s , 'cmor3.0-1.0')

#μορφη: pywt.cwt( data = input signal , scales , wavelet ) 

print(coefs.shape)

#2.1.d

#%matplotlib inline
t = np.linspace(0/1000,2000/1000,2000)
f = freqs*1000
plt.figure(8)
plt.pcolormesh(t,f,np.abs(coefs))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('CWT of w1')
plt.show()

plt.figure(9)
plt.pcolormesh(t,s,np.abs(coefs))
plt.xlabel('Time (sec)')
plt.ylabel('s')
plt.title('CWT of w1')

plt.show()