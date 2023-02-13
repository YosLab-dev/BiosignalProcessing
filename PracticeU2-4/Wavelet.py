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
        [cA1, cD1, cD2, cD3, cD4, cD5, cD6, cD7, cD8, cD9, cD10,
         cD11, cD12, cD13, cD14, cD15, cD16, cD17, cD18, cD19, cD20,
         cD21, cD22, cD23, cD24, cD25] = pywt.wavedec(signal[l], 'db4', level=25)
        getStats(cA1, results, row, 0)
        getStats(cD1, results, row, 6)
        getStats(cD2, results, row, 12)
        getStats(cD3, results, row, 18)
        getStats(cD4, results, row, 24)
        getStats(cD5, results, row, 30)
        getStats(cD6, results, row, 36)
        getStats(cD7, results, row, 42)
        getStats(cD8, results, row, 48)
        getStats(cD9, results, row, 54)
        getStats(cD10, results, row, 60)
        getStats(cD11, results, row, 66)
        getStats(cD12, results, row, 72)
        getStats(cD13, results, row, 78)
        getStats(cD14, results, row, 84)
        getStats(cD15, results, row, 90)
        getStats(cD16, results, row, 96)
        getStats(cD17, results, row, 102)
        getStats(cD18, results, row, 108)
        getStats(cD19, results, row, 114)
        getStats(cD20, results, row, 120)
        getStats(cD16, results, row, 126)
        getStats(cD17, results, row, 132)
        getStats(cD18, results, row, 138)
        getStats(cD19, results, row, 144)
        getStats(cD20, results, row, 150)
        row += 1


#    print(results)
#cD21, cD22, cD23, cD24, cD25, cD26, cD27, cD28, cD29, cD30,
        # cD31, cD32

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
ARRChar = np.zeros((10, 156))
CHFChar = np.zeros((10, 156))
NSRChar = np.zeros((10, 156))
allvalues = np.zeros((30, 156))

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
    'Mean cD8', 'Std cD8', 'Min cD8', 'Max cD8', 'TWE cD8', 'IWE cD8',
    'Mean cD9', 'Std cD9', 'Min cD9', 'Max cD9', 'TWE cD9', 'IWE cD9',
    'Mean cD10', 'Std cD10', 'Min cD10', 'Max cD10', 'TWE cD10', 'IWE cD10',
    'Mean cD11', 'Std cD11', 'Min cD11', 'Max cD11', 'TWE cD11', 'IWE cD11',
    'Mean cD12', 'Std cD12', 'Min cD12', 'Max cD12', 'TWE cD12', 'IWE cD12',
    'Mean cD13', 'Std cD13', 'Min cD13', 'Max cD13', 'TWE cD13', 'IWE cD13',
    'Mean cD14', 'Std cD14', 'Min cD14', 'Max cD14', 'TWE cD14', 'IWE cD14',
    'Mean cD15', 'Std cD15', 'Min cD15', 'Max cD15', 'TWE cD15', 'IWE cD15',
    'Mean cD16', 'Std cD16', 'Min cD16', 'Max cD16', 'TWE cD16', 'IWE cD16',
    'Mean cD17', 'Std cD17', 'Min cD17', 'Max cD17', 'TWE cD17', 'IWE cD17',
    'Mean cD18', 'Std cD18', 'Min cD18', 'Max cD18', 'TWE cD18', 'IWE cD18',
    'Mean cD19', 'Std cD19', 'Min cD19', 'Max cD19', 'TWE cD19', 'IWE cD19',
    'Mean cD20', 'Std cD20', 'Min cD20', 'Max cD20', 'TWE cD20', 'IWE cD20',
    'Mean cD21', 'Std cD21', 'Min cD21', 'Max cD21', 'TWE cD21', 'IWE cD21',
    'Mean cD22', 'Std cD22', 'Min cD22', 'Max cD22', 'TWE cD22', 'IWE cD22',
    'Mean cD23', 'Std cD23', 'Min cD23', 'Max cD23', 'TWE cD23', 'IWE cD23',
    'Mean cD24', 'Std cD24', 'Min cD24', 'Max cD24', 'TWE cD24', 'IWE cD24',
    'Mean cD25', 'Std cD25', 'Min cD25', 'Max cD25', 'TWE cD25', 'IWE cD25'])

dataFrameValues.to_csv("CharacteristicsCoeff.csv")