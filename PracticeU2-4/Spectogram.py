import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import signal
import pandas as pd


def showInstances(instance, signals, tI, tF, step, title):
    j = 1
    time = np.linspace(tI, tF, step)
    for i in instance:
        plt.suptitle(title, fontsize=14)
        plt.subplot(10, 1, j)
        plt.plot(time, signals[i])
        plt.ylabel(str(i))
        plt.xlabel("Time [s]")
        j = j + 1
    plt.show()


def getSample(vector, signal, start, stop):
    for i in range(0, len(signal), 1):
        vector[i] = signal[i, start:stop]


def getSTFT_FFT(vector, typeSignal, results, tFourier):
    m = 0
    for l in vector:

        #Calculating the STFT
        f, t, Zxx = signal.stft(typeSignal[l], 128, nperseg=64, scaling='spectrum')
        ZxxAbs = np.abs(Zxx)
        dataF = pd.DataFrame(ZxxAbs).T
        charDF = dataF.describe()
        timeARR = np.linspace(1000, 1010, 1280)

        #Generate the frequency axis
        n = 2560
        freq = (128/2)*np.linspace(0,1,int(n/2))
        #Obtaining the coefficients of the Fourier Transform
        x = np.fft.fft(ARR10sec[l])
        tFourier = (10/n)*np.abs(x[0:np.size(freq)]) #
        f1, t1, zx1 = signal.stft(tFourier, 128, nperseg=64, scaling='spectrum')
        # print(charDF)
        k = 0
        for i in range(0, 33, 1):
            for j in range(1, 8, 1):
                results[m, k] = charDF.iloc[j, i].T
                k += 1
        m += 1


        plt.subplot(3, 1, 1)
        plt.title("MIT-BIH Normal Sinus Rhythm Signal and Spectrogram: " +str(l))
        plt.plot(timeARR, typeSignal[l])
        plt.ylabel('Magnitude')
        plt.xlabel('Time [sec]')
        plt.subplot(3, 1, 2)
        plt.pcolormesh(t, f, np.abs(Zxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.subplot(3, 1, 3)
        plt.plot(freq, tFourier)
        plt.tight_layout()
        plt.ylabel('Magnitude (Normalized)')
        plt.xlabel('Frequency [Hz]')
        plt.show()


# Reading the file .mat
matFile = loadmat('ECGValues.mat')
data = matFile['Data']  # Selecting only the variable that contains the channel data

ARR10sec = np.zeros((96, 1280))  # This will store a 10 second sample of all instances
CHF10sec = np.zeros((30, 1280))
NSR10sec = np.zeros((36, 1280))
ARRChar = np.zeros((10, 231))
CHFChar = np.zeros((10, 231))
NSRChar = np.zeros((10, 231))
ARRFFT = np.zeros((10, 231))
CHFFFT = np.zeros((10, 231))
NSRFFT = np.zeros((10, 231))

# Separating the channel between the three classes
ARR = data[0:96]  # MIT-BIH Arrhythmia Database
CHF = data[96:126]  # BIDMC Congestive Heart Failure Database
NSR = data[126:163]  # MIT-BIH Normal Sinus Rhythm Database

# Defining random vectors of instances
vARR = [2, 18, 22, 36, 48, 59, 73, 80, 86, 95]
vCHF = [3, 7, 11, 12, 18, 21, 22, 24, 27, 29]
vNSR = [0, 10, 13, 14, 17, 18, 20, 22, 24, 32]

# Extracting 10 seconds of data sample
getSample(ARR10sec, ARR, 1499, 2779)
getSample(NSR10sec, NSR, 2499, 3779)
getSample(CHF10sec, CHF, 999, 2279)


#getSTFT_FFT(vARR, ARR10sec, ARRChar, ARRFFT)
#getSTFT_FFT(vCHF, CHF10sec, CHFChar, CHFFFT)
getSTFT_FFT(vNSR, NSR10sec, NSRChar, NSRFFT)
#values = np.concatenate((ARRChar, CHFChar, NSRChar), axis=0)


"""
dF = pd.DataFrame(values, columns=[
    'Mean [0 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [2 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [4 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [6 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [8 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [10 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [12 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [14 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [16 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [18 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [20 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [22 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [24 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [26 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [28 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [30 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [32 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [34 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [36 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [38 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [40 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [42 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [44 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [46 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [48 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [50 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [52 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [54 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [56 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [58 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [60 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [62 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
    'Mean [64 Hz]', 'Std', 'Min', '25%', '50%', '75%', 'Max',
])
dF.to_csv("AllCharacteristics.csv")
"""