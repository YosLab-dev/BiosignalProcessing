import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.io import loadmat
from scipy import signal
import pandas as pd
import math


def getSample(vector, signal, start, stop):
    for i in range(0, len(signal), 1):
        vector[i] = signal[i, start:stop]

def getTWE_IWE(vector, signal, results):
    j = 0
    row = 0
    for l in vector:
        [cA1, cD1, cD2, cD3, cD4, cD5, cD6, cD7, cD8] = pywt.wavedec(signal[l], 'db4', level=8)
        plt.subplot(4,1,1)
        plt.title("Discrete Wavelet Transform coefficients at each level for the channel:"+str(l))
        plt.stem(cD1, linefmt='tomato')
        plt.stem(cD2, linefmt='darkorange')
        plt.legend(['Level 1', 'Level 2'])
        plt.xlabel('Coefficients')
        plt.ylabel('Amplitude')
        plt.subplot(4, 1, 2)
        plt.stem(cD3, linefmt='yellowgreen')
        plt.stem(cD4, linefmt='turquoise')
        plt.legend(['Level 3', 'Level 4'])
        plt.xlabel('Coefficients')
        plt.ylabel('Amplitude')
        plt.subplot(4, 1, 3)
        plt.stem(cD5, linefmt='dodgerblue')
        plt.stem(cD6, linefmt='slateblue')
        plt.legend(['Level 5', 'Level 6'])
        plt.xlabel('Coefficients')
        plt.ylabel('Amplitude')
        plt.subplot(4, 1, 4)
        plt.stem(cD7, linefmt='darkmagenta')
        plt.stem(cD8, linefmt='deeppink')
        plt.legend(['Level 7', 'Level 8'])
        plt.xlabel('Coefficients')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
        getStats(cA1, results, row, 0)
        getStats(cD1, results, row, 6)
        getStats(cD2, results, row, 12)
        getStats(cD3, results, row, 18)
        getStats(cD4, results, row, 24)
        getStats(cD5, results, row, 30)
        getStats(cD6, results, row, 36)
        getStats(cD7, results, row, 42)
        getStats(cD8, results, row, 48)
        row += 1




def getStats(coefficients, results, row, column):
    results[row, column] = np.mean(coefficients)
    column += 1
    results[row, column] = np.std(coefficients)
    column += 1
    results[row, column] = np.min(coefficients)
    column += 1
    results[row, column] = np.max(coefficients)
    column += 1
    results[row, column], results[row, column + 1] = TWE_IWE(coefficients)


def TWE_IWE(coefficients):
    N = len(coefficients)
    tweS = 0
    iweS = 0
    for i in range(1, N - 1, 1):
        tweS += np.abs(coefficients[i] ** 2 - coefficients[i - 1] * coefficients[i + 1])
        iweS += coefficients[i - 1] ** 2
    TWE = math.log((tweS / N), 10)
    IWE = math.log((iweS / N), 10)
    return TWE, IWE


# Reading the file .mat
matFile = loadmat('ECGValues.mat')
data = matFile['Data']  # Selecting only the variable that contains the channel data

ARR10sec = np.zeros((96, 1280))  # This will store a 10 second sample of all instances
CHF10sec = np.zeros((30, 1280))
NSR10sec = np.zeros((36, 1280))
ARRChar = np.zeros((10, 54))
CHFChar = np.zeros((10, 54))
NSRChar = np.zeros((10, 54))
allvalues = np.zeros((30, 54))

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


getTWE_IWE(vARR, ARR10sec, ARRChar)
getTWE_IWE(vNSR, NSR10sec, NSRChar)
getTWE_IWE(vCHF, CHF10sec, CHFChar)

allvalues = np.concatenate((ARRChar, CHFChar, NSRChar))
dataFrameValues = pd.DataFrame(allvalues, columns=[
    'Mean cA1', 'Std cA1', 'Min cA1', 'Max cA1', 'TWE cA1', 'IWE cA1',
    'Mean cD1', 'Std cD1', 'Min cD1', 'Max cD1', 'TWE cD1', 'IWE cD1',
    'Mean cD2', 'Std cD2', 'Min cD2', 'Max cD2', 'TWE cD2', 'IWE cD2',
    'Mean cD3', 'Std cD3', 'Min cD3', 'Max cD3', 'TWE cD3', 'IWE cD3',
    'Mean cD4', 'Std cD4', 'Min cD4', 'Max cD4', 'TWE cD4', 'IWE cD4',
    'Mean cD5', 'Std cD5', 'Min cD5', 'Max cD5', 'TWE cD5', 'IWE cD5',
    'Mean cD6', 'Std cD6', 'Min cD6', 'Max cD6', 'TWE cD6', 'IWE cD6',
    'Mean cD7', 'Std cD7', 'Min cD7', 'Max cD7', 'TWE cD7', 'IWE cD7',
    'Mean cD8', 'Std cD8', 'Min cD8', 'Max cD8', 'TWE cD8', 'IWE cD8'
   ])

dataFrameValues.to_csv("CharacteristicsCoeff.csv")
