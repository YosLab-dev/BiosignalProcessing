
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


#Subject 1
file_H = loadmat('emg_healthym.mat')
Healthy = np.transpose(file_H['val'])

file_N = loadmat('emg_neuropathym.mat')
Neuroph = np.transpose(file_N['val'])

file_M = loadmat('emg_myopathym.mat')
Myopath = np.transpose(file_M['val'])


#Getting the stadistical parameters 
Healthy_Mean = np.mean(Healthy)
Healthy_Dev = np.std(Healthy)
Healthy_Max = np.max(Healthy)
Healthy_Min = np.min(Healthy)

Neuroph_Mean = np.mean(Neuroph)
Neuroph_Dev = np.std(Neuroph)
Neuroph_Max = np.max(Neuroph)
Neuroph_Min = np.min(Neuroph)

Myopath_Mean = np.mean(Myopath)
Myopath_Dev = np.std(Myopath)
Myopath_Max = np.max(Myopath)
Myopath_Min = np.min(Myopath)

#Signal preprocessing

for n in Healthy:
    Healthy_S = (Healthy - Healthy_Mean)/Healthy_Dev
    
    
for n in Neuroph:
    Neuroph_S = (Neuroph - Neuroph_Mean)/Neuroph_Dev
    
    
for n in Myopath:
    Myopath_S = (Myopath - Myopath_Mean)/Myopath_Dev
    
for n in Healthy:
    Healthy_N = (Healthy - Healthy_Min)/(Healthy_Max - Healthy_Min)

for n in Neuroph:
    Neuroph_N = (Neuroph - Neuroph_Min)/(Neuroph_Max - Neuroph_Min)
    
for n in Myopath:
    Myopath_N = (Myopath - Myopath_Min)/(Myopath_Max - Myopath_Min)
    
t = np.linspace(0, 10, np.size(Healthy))

plt.rcParams["figure.figsize"] = (20,5)

plt.plot(t, Healthy)
plt.title("EMG Healthy")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("HealthyO.png", dpi = 600)
plt.show()


plt.plot(t, Neuroph)
plt.title("EMG Neuropathy")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("NeuropathyO.png", dpi = 600)
plt.show()

plt.plot(t, Myopath)
plt.title("EMG Myopathy")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("MyopathyO.png", dpi = 600)
plt.show()

plt.plot(t, Healthy_S)
plt.title("EMG Healthy (Standarized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("HealthyS.png", dpi = 600)
plt.show()

plt.plot(t, Neuroph_S)
plt.title("EMG Neuropathy (Standarized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("NeuropathyS.png", dpi = 600)
plt.show()

plt.plot(t, Myopath_S)
plt.title("EMG Myopathy (Standarized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("MyopathyS.png", dpi = 600)
plt.show()

plt.plot(t, Healthy_N)
plt.title("EMG Healthy (Normalized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("HealthyN.png", dpi = 600)
plt.show()

plt.plot(t, Neuroph_N)
plt.title("EMG Neuropathy (Normalized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("NeuropathyN.png", dpi = 600)
plt.show()

plt.plot(t, Myopath_N)
plt.title("EMG Myopathy (Normalized)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig("MyopathyN.png", dpi = 600)
plt.show()


"""

plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(x, y)
plt.title("y=mx+c")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
"""
