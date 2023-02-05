import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statistics as std
from scipy.io import loadmat
from scipy import signal
import librosa
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from seaborn import load_dataset

def showInstances(instance, signals, tI, tF, step, title):
    j = 1
    time = np.linspace(tI, tF, step)
    for i in instance:
        plt.suptitle(title, fontsize = 14)
        plt.subplot(10, 1, j)
        plt.plot(time, signals[i])
        plt.ylabel(str(i))
        plt.xlabel("Time [s]")
        j = j + 1
    plt.show()

def getSample(vector, signal, start, stop):
    for i in range (0, len(signal), 1):
        vector[i] = signal[i, start:stop]

def parametersExtration(signal, char):

    for i in range(0, 10, 1):  # Channel selection
        j = 0
        # Analyzing the signal in channel 'i'
        signalMin = np.min(signal[i])
        char[i,j] = signalMin
        j = j + 1  #

        signalMax = np.max(signal[i])
        char[i,j] = signalMax
        j = j + 1

        signalMean = np.mean(signal[i])
        char[i,j] = signalMean
        j = j + 1

        signalRMS = np.sqrt(np.mean(signal[i] ** 2))
        char[i,j] = signalRMS
        j = j + 1

        signalV = std.variance(signal[i])
        char[i,j] = signalV
        j = j + 1

        signalDeviation = std.stdev(signal[i])
        char[i,j] = signalDeviation
        j = j + 1

        signalK = scipy.stats.kurtosis(signal[i])
        char[i,j] = signalK
        j = j + 1

        signalSk = scipy.stats.skew(signal[i])
        char[i,j] = signalSk
        j = j + 1

        signalEnt = scipy.stats.entropy(signal[i])
        char[i,j] = signalEnt
        j = j + 1

        signalMed = std.median(signal[i])
        char[i,j] = signalMed
        j = j + 1

        signalZCR = np.mean(librosa.feature.zero_crossing_rate(signal[i]))
        char[i,j] = signalZCR
        j = j + 1

        signalMCR = np.mean(librosa.feature.zero_crossing_rate(signal[i] - signalMean))
        char[i,j] = signalMCR
        j = j + 1

def writingXLSX(signal,name):
    data = pd.DataFrame(signal, columns = ['Min', 'Max', 'Mean', 'RMS', 'Variance', 'Std.Dev.', 'Kurtosis', 'Skewness', 'Entropy', 'Median', 'ZCR', 'MCR'])
    dataExc = pd.ExcelWriter(str(name)+".xlsx")
    data.to_excel(dataExc, sheet_name = str(name))
    dataExc.save()

#Reading the file .mat
matFile = loadmat('ECGValues.mat')
data = matFile['Data'] #Selecting only the variable that contains the channel data

ARR10sec = np.zeros((96, 1280)) #This will store a 10 second sample of all instances
CHF10sec = np.zeros((30, 1280))
NSR10sec = np.zeros((36, 1280))
char = np.zeros((10,12))
ARR10secStd = np.zeros((96, 1280))
CHF10secStd = np.zeros((30, 1280))
NSR10secStd = np.zeros((36, 1280))


#Separating the channel between the three classes
ARR = data[0:96] #MIT-BIH Arrhythmia Database
CHF = data[96:126] #BIDMC Congestive Heart Failure Database
NSR = data[126:163] #MIT-BIH Normal Sinus Rhythm Database

#Defining random vectors of instances
vARR = [2, 18, 22, 36, 48, 59, 73, 80, 86, 95]
vCHF = [3, 7, 11, 12, 18, 21, 22, 24, 27, 29]
vNSR = [0, 10, 13, 14, 17, 18, 20, 25, 28, 32]
samples = np.linspace(0, np.size(data[0]),np.size(data[0]))

#Extracting 10 seconds of data sample
getSample(ARR10sec, ARR, 1499, 2779)
getSample(NSR10sec, NSR, 2499, 3779)
getSample(CHF10sec, CHF, 999, 2279)

parametersExtration(ARR10sec, char)
writingXLSX(char,"ARRParameters")

parametersExtration(NSR10sec, char)
writingXLSX(char,"NSRParameters")

parametersExtration(CHF10sec, char)
writingXLSX(char,"CHFParameters")

#showInstances(vARR, ARR, 0, 3600, 65536, "MIT-BIH Arrhythmia Database")
#showInstances(vNSR, NSR, 0, 3600, 65536, "MIT-BIH Normal Sinus Rhythm Database")
#showInstances(vCHF, CHF, 0, 3600, 65536, "BIDMC Congestive Heart Failure Database")

#showInstances(vCHF, CHF10sec, 999, 1009, 1280, "BIDMC Congestive Heart Failure Sample")
#showInstances(vARR, ARR10sec, 1499, 1509, 1280, "MIT-BIH Arrhythmia Sample")
#showInstances(vNSR, NSR10sec, 2499, 2509, 1280, "MIT-BIH Normal Sinus Rhythm Sample")


dfARR = pd.read_excel("ARRParameters.xlsx")
dfNSR = pd.read_excel("NSRParameters.xlsx")
dfCHF = pd.read_excel("CHFParameters.xlsx")

plt.title("RMS comparison")
plt.grid(linewidth = 0.3)
plt.hist(dfNSR['RMS'], bins = 7, cumulative = False, density = True, label = "NSR")
plt.hist(dfARR['RMS'], bins = 7, cumulative = False, density = True, label = "ARR")
plt.hist(dfCHF['RMS'], bins = 7, cumulative = False, density = True, label = "CHF")
plt.legend()
plt.show()

plt.title("ZCR comparison")
plt.grid(linewidth = 0.3)
plt.hist(dfARR['ZCR'], bins = 7, cumulative = False, density = True, label = "ARR")
plt.hist(dfNSR['ZCR'], bins = 7, cumulative = False, density = True, label = "NSR")
plt.hist(dfCHF['ZCR'], bins = 7, cumulative = False, density = True, label = "CHF")
plt.legend()
plt.show()

plt.title("MCR comparison")
plt.grid(linewidth = 0.3)
plt.hist(dfARR['MCR'], bins = 7, cumulative = False, density = True, label = "ARR")
plt.hist(dfNSR['MCR'], bins = 7, cumulative = False, density = True, label = "NSR")
plt.hist(dfCHF['MCR'], bins = 7, cumulative = False, density = True, label = "CHF")
plt.legend()
plt.show()
