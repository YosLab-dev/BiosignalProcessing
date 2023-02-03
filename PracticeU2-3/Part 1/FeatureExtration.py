
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statistics as std
from scipy.io import loadmat
from scipy import signal
import librosa

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

#Reading the file .mat
data = loadmat('mydata.mat')

signals = data['dt'] #Selecting only the varibale that contains the channel data
signalsT = np.transpose(signals) #Transposing the matrix to separate each channel
t = np.linspace(0, np.size(signalsT[0]),np.size(signalsT[0]))
char = np.zeros(48) #In this vector we're going to save all the features

#This variables are only for plotting purposes
RMS2 = np.zeros(2500)
RMS4 = np.zeros(2500)
RMS6 = np.zeros(2500)
RMS8 = np.zeros(2500)
charT = char
j = 0

#Dimentions: signals 2500x8, signalsT 8x2500
#Channels: 1, 3, 5, 7

for i in range(1, 8, 2): #Channel selection
    # Analyzing the signal in channel 'i'
    signalMin = np.min(signalsT[i])
    char[j] = signalMin
    j = j + 1 #

    signalMax = np.max(signalsT[i])
    char[j] = signalMax
    j = j + 1

    signalMean = np.mean(signalsT[i])
    char[j] = signalMean
    j = j + 1

    signalRMS = np.sqrt(np.mean(signalsT**2))
    char[j] = signalRMS
    j = j + 1

    signalV = std.variance(signalsT[i])
    char[j] = signalV
    j = j + 1

    signalDeviation = std.stdev(signalsT[i])
    char[j] = signalDeviation
    j = j + 1

    signalK = scipy.stats.kurtosis(signalsT[i])
    char[j] = signalK
    j = j + 1

    signalSk = scipy.stats.skew(signalsT[i])
    char[j] = signalSk
    j = j + 1

    signalEnt = scipy.stats.entropy(signalsT[i])
    char[j] = signalEnt
    j = j + 1

    signalMed = std.median(signalsT[i])
    char[j] = signalMed
    j = j + 1

    signalZCR = np.mean(librosa.feature.zero_crossing_rate(signalsT[i]))
    char[j] = signalZCR
    j = j + 1

    signalMCR = np.mean(librosa.feature.zero_crossing_rate(signalsT[i]-signalMean))
    char[j] = signalMCR
    j = j + 1



for i in range(0,2500,1):
    RMS2[i] = char[2]
    RMS4[i] = char[14]
    RMS6[i] = char[26]
    RMS8[i] = char[38]

for j in range(0,48,1):
    charT[j] = truncate(char[j], 2)

print(charT)

"""
plt.subplot(4, 1, 1)
plt.title("Original Signals")
plt.plot(t, signalsT[1])
plt.ylabel("Channel 2")

plt.subplot(4, 1, 2)
plt.plot(t, signalsT[3], color = "orange")
plt.ylabel("Channel 4")

plt.subplot(4, 1, 3)
plt.plot(t, signalsT[5], color = "yellowgreen")
plt.ylabel("Channel 6")

plt.subplot(4,1,4)
plt.plot(t,signalsT[7], color = "darkturquoise")
plt.ylabel("Channel 8")
plt.xlabel("Samples")
#plt.savefig("OriginalSignals.png", dpi = 600)
#plt.show()
"""
plt.subplot(4, 1, 1)
plt.title("Mean Cross Rate Comparison")
plt.plot(t, signalsT[1])
plt.plot(t, RMS2, color = "red", label = 'RMS = '+str(truncate(char[2],2)))
plt.legend()
plt.ylabel("Channel 2")

plt.subplot(4, 1, 2)
plt.plot(t, signalsT[3], color = "orange")
plt.plot(t, RMS4, color = "red", label = 'RMS = '+str(truncate(char[14],2)))
plt.legend()
plt.ylabel("Channel 4")

plt.subplot(4, 1, 3)
plt.plot(t, signalsT[5], color = "yellowgreen")
plt.plot(t, RMS6, color = "red", label = 'RMS = '+str(truncate(char[26],2)))
plt.legend()
plt.ylabel("Channel 6")

plt.subplot(4,1,4)
plt.plot(t,signalsT[7], color = "darkturquoise")
plt.plot(t, RMS8, color = "red", label = 'RMS = '+str(truncate(char[38],2)))
plt.legend()
plt.ylabel("Channel 8")
plt.xlabel("Samples")
plt.show()
