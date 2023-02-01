
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import signal

# Reading the file .mat
data = loadmat('mydata.mat')

signals = data['dt']  # Selecting only the variable that contains the channel data
signalsT = np.transpose(signals)  # Transposing the matrix to separate each channel
Acc = 0
instMeanV = np.zeros(2500)
t = np.zeros(2500)
CAR_Channel = np.zeros((8, 2500))
for i in range(0,2500,1): #Instant of time
    for j in range(0,8,1): #Each channel
        Acc = signals[i,j] + Acc
    instMeanV[i] = Acc/8
    Acc = 0

#Calculating the CAR
for i in range(0,2500,1):
    t[i] = i
    for j in range(0,8,1):
        CAR_Channel[j,i] = signals[i,j] - instMeanV[j]


plt.subplot(8,1,1)
plt.title("CAR method applied to each signal")
for i in range (0,8,1):
    plt.subplot(8,1,i + 1)
    plt.plot(t,CAR_Channel[i])
    plt.ylabel("Ch: " +str(i) )
plt.savefig("CAR_Method.png", dpi = 600)
plt.show()

plt.subplot(8,1,1)
plt.title("Original Signals")
for i in range (0,8,1):
    plt.subplot(8,1,i + 1)
    plt.plot(t,signalsT[i])
    plt.ylabel("Ch: " +str(i) )
plt.savefig("Original_Signals.png", dpi = 600)
plt.show()