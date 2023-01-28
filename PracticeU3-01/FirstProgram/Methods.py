
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import signal

#Reading the file .mat
data = loadmat('mydata.mat')

signals = data['dt'] #Selecting only the varibale that contains the channel data
signals = np.transpose(signals) #Transposing the matrix to separate each channel

for i in range(0,8,1):
    t = np.linspace(0, np.size(signals[0]),np.size(signals[0]))
    channel = signals[i]
    
    #Analyzing the signal in channel 'i'
    signalMean = np.mean(channel)
    #print(signalMean)
    signalDeviation = np.std(channel)
    #print(signalDeviation)
    signalMax = np.max(channel)
    signalMin = np.min(channel)
    
    #Standarization
    for n in channel:
        S = (channel-signalMean)/signalDeviation

    #Normalization
    for n in channel:
        N = (channel-signalMin)/(signalMax-signalMin)

    #Detrending
    D = signal.detrend(channel)

    #Common Average Reference
    for n in channel:
        CAR = channel-signalMean
        
    #Showing all the signals
    plt.plot(t,channel)
    plt.title("Original Signal Channel:"+str(i))
    plt.savefig("OriginalChannel"+str(i)+".png", dpi = 600)
    plt.show()


    plt.plot(t,S)
    plt.title("Standardized Signal Channel:"+str(i))
    plt.savefig("StandardizedChannel"+str(i)+".png", dpi = 600)
    plt.show()

    plt.plot(t,N)
    plt.title("Normalized Signal Channel:"+str(i))
    plt.savefig("NormalizedChannel"+str(i)+".png", dpi = 600)
    plt.show()

    plt.plot(t,D)
    plt.title("Detrendign Signal Channel:"+str(i))
    plt.savefig("DetrendignChannel"+str(i)+".png", dpi = 600)
    plt.show()

    plt.plot(t,CAR)
    plt.title("CAR Signal Channel:"+str(i))
    plt.savefig("CARChannel"+str(i)+".png", dpi = 600)
    plt.show()
