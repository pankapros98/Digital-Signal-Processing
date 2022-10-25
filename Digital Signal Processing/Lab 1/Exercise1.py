import os
import numpy as np
import scipy as sp
import librosa
import matplotlib.pyplot as plt
import pywt
import sounddevice as sd
import time
import soundfile as sf
from scipy import signal
from scipy.signal import argrelextrema

##### FUNCTION SECTION #####

# 1.1

# Function that creatd the 'd' sound samples

def make_sound_samples(row, line):
    d = np.sin(line*n) + np.sin(row*n)
    return d

# 1.2

# Caculating the DFT of d2 and d7 sound sample => that's numbers '2' and '7'
# I will make this process more abstract by creating a function that you insert which number you want to calculate

def choose_sound_sample(inputs):
    return d[inputs]

def abs_fft_calc(sound_sample):
    # calculating |fft|
    result =abs(np.fft.fft(d[sound_sample]))
    
    plt.figure(figsize = (14, 5))
    plt.plot(np.abs(result))
    plt.xlabel('Frequency')
    plt.ylabel('DFT for Sound Signal No: ' + str(sound_sample))
    plt.show()
    return result

# 1.3

# I will create a function that accepts all input numbers and will create the final sound sample and file

def create_sound_file(input_numbers):
    zeros = np.zeros(100)   # Exercise requests 100 zeros
    tone_sequence = []
    AM = input_numbers
    tone_sequence = np.array([])

    for i in range(0, len(AM)):
        tone_sequence = np.concatenate((tone_sequence, choose_sound_sample(AM[i])), axis = None)
        tone_sequence = np.concatenate((tone_sequence, zeros), axis = None)


   # librosa.output.write.wav('tone_sequence.wav', tone_sequence, 8192, 1)
    sf.write('tone_sequence.wav', tone_sequence, 8192, 'PCM_24')

    plt.figure(figsize = (14, 5))
    plt.plot(tone_sequence)
    plt.title('Tone Sequence Graph')
    plt.show()
    return tone_sequence


# 1.4

def windows_calcs(input_numbers):

    #rectangular windows
    rect_win = signal.boxcar(1000)

    #Hamming windows
    ham_win = signal.hamming(1000)
    
    #helping matrices
    space = []
    space_rect = []

    #calling the function create_sound_file() in order to get "tone_sequence" array and copy it in tone_sequence_ arrauy

    tone_sequence_ = create_sound_file(input_numbers)

    for i in range(0, len(input_numbers)):
    	#for hamming windows
        space = tone_sequence_[(i * 1000 + i * 100):((i + 1) * 1000 + i * 100)]
        space = ham_win * space

        #for rectangular windows
        space_rect = tone_sequence_[(i * 1000 + i * 100):((i + 1) * 1000 + i * 100)]
        space_rect = rect_win * space_rect

        #creating 1000 samples
        f = np.linspace(0,2*np.pi, 1000)
        y = np.fft.fft(space)
        y_rect = np.fft.fft(space_rect)

        plt.figure(figsize = (14, 5))
        plt.plot(f, np.abs(y))
        
        plt.figure(figsize = (14, 5))
        plt.plot(f, np.abs(y_rect))
        
    plt.show()

# 1.5

# Function that creates the 'k' list

def k_fr_list():
    #creating an index matrix
    index = [[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]

    for i in range(len(index)):
        y = np.abs(np.fft.fft(d[i]))**2
        plt.plot(np.abs(y))

        #save the biggest samples here
        ar = []
        ar = np.sort(np.argpartition(y[0:499], -2)[-2:])
        index[i][0] = ar[0]
        index[i][1] = ar[1]
        print('Pointers at: ' + str(i) + ' : ' + str(index[i][0]) + ' & ' + str(index[i][1]) + ' and frequencies ' + str(2*np.pi*index[i][0]/1000) + ' & ' + str(2*np.pi*index[i][1]/1000))

    return index


# 1.6

# Function that finds the number that is represented by the info provided by function 'ttdecode'

def kick_peak(en):
    index = k_fr_list()
    re = []
    re = (np.sort(np.argpartition(en[0:500], -2)[-2:]))

    for j in range(0,10):
        if (np.all(index[j] == re)):
            break
    return j


def ttdecode(y):
    vector = []
    window=1000
    i = 0
    length = len(y)

    while (i < (length)):

        # while i is within limits of the matrix boundaries
        while ((y[i]==0) and (i < (length))):
            
            # if the sample of the matrix is '0' we ignore it
            i = i + 1

        # when the sample of the matrix is not '0' we take the 1000 sample matrix
        # we find the fft of that and then we call function kick_peak to find the number we desire to find
        energy = []
        energy = np.abs(np.fft.fft(y[i:i+1000]))**2
        #ar = np.sort(np.argpartition(energy[0:500], -2)[-2:])

        # finally we put the number we found at the vector to store it
        vector.append(kick_peak(energy))

        # go to the next window 
        i+=1000

    return vector


##### Function Calling Section #####

#1.1

# make the samples
n = np.linspace(0,1000,1000)

# matrices that help us to build the 'd' samples
lines = np.array([0.9273, 1.0247, 1.1328])
rows = np.array([0.5346, 0.5906, 0.6535])       #I exclude the row for the '0' for use at it's own 
d = []

# make the '0' sound sample
d.append(np.sin(lines[1]*n) + np.sin(0.7217*n))

# create the sound samples '1' - '9'
for i in rows:
    for j in lines:
        d.append(make_sound_samples(i,j))

# play the sound samples
for i in d:
    sd.play(i,8192)
    time.sleep(0.8)
plt.show()


# plot the sound samples
for i in range(len(d)):
    plt.figure(figsize = (14, 5))
    plt.plot(d[i])
    plt.xlabel('Sound Samples n')
    plt.ylabel('Sound Signal No: ' + str(i))
    plt.show()


# 1.2

# Calculate |fft| of number '2' and '7' by callinf the function abs_fft_calc
abs_fft_calc(2)
abs_fft_calc(7)

# 1.3

# The input array is the sum of our AMs => 03118926 + 03118868 and then create the sound file 'tone_sequence.wav'
input_num = [0, 6, 2, 3, 7, 7, 9, 4]
create_sound_file(input_num)

# 1.4

# call the function windows calcs using the numbers given in 1.3
windows_calcs(input_num)

# 1.5

# calling function k_fr_list for creating the k list
k_fr_list()


# 1.6

file_selected = 'tone_sequence.wav'
tone_call, sr = librosa.core.load(file_selected, sr = 8192) # you have to be on the programm's path in order to find the file 
                                                            # or else change the name of the file_selected with the path of your file you want to find
tone_signal = [None]*8802                                   # 8802 used ONLY for our tone_sequence file

for i in range(8800):                                       # range(8800) used ONLY for the 'tone_sequence.wav' file
    tone_signal[i] = tone_call[i]

print('1.6: ' + str(ttdecode(tone_signal)))


# 1.7

easySig = np.load(r'easySig.npy')
hardSig = np.load(r'hardSig.npy')

print('1.7 easySig: ' + str(ttdecode(easySig)))
print('1.7 hardSig: ' + str(ttdecode(hardSig)))

