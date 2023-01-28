
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy import signal
from scipy.fft import fft, fftfreq


sound = 'ecg.wav'
sampling, data = waves.read(sound)
samples = len(data)
T = 1/4000
t = np.arange(0, samples/sampling, 1/sampling)
x = np.linspace(0.0, samples*T, samples, endpoint=False)
x_FFT = fftfreq(samples, T)[:samples//2]
ECG_FFT = fft(data)
ecgMax = np.max(data)
ecgMin = np.min(data)
order = 2


#Defining the functions for each filter

def lowPassFilter(cutoff, sampling, order):
    NyqF = sampling/2
    cutoffNormal = cutoff/NyqF
    b, a = signal.butter(order, cutoffNormal, btype = 'lowpass', analog = False)
    return b, a

def highPassFilter(cutoff, sampling, order):
    NyqF = sampling/2
    cutoffNormal = cutoff/NyqF
    b, a = signal.butter(order, cutoffNormal, btype = 'highpass', analog = False)
    return b, a

def bandPassFilter(lowC, highC, sampling, order):
    NyqF = sampling/2
    lowCutoffNormal = lowC/NyqF
    highCutoffNormal = highC/NyqF
    b, a = signal.butter(order, [lowCutoffNormal,highCutoffNormal], btype = 'bandpass', analog = False)
    return b, a

def bandStopFilter(lowC, highC, sampling, order):
    NyqF = sampling/2
    lowCutoffNormal = lowC/NyqF
    highCutoffNormal = highC/NyqF
    b, a = signal.butter(order, [lowCutoffNormal,highCutoffNormal], btype = 'bandstop', analog = False)
    return b, a


#Applying the filters
b1,a1 = lowPassFilter(60, sampling, order)
LP_Filtered = signal.filtfilt(b1, a1, data)
LP_FFT = fft(LP_Filtered) #Analyzing in the frequency domain

b2,a2 = highPassFilter(30, sampling, order)
HP_Filtered = signal.filtfilt(b2, a2, data)
HP_FFT = fft(HP_Filtered)

b3,a3 = bandPassFilter(2, 50, sampling, order)
BP_Filtered = signal.filtfilt(b3, a3, data)
BP_FFT = fft(BP_Filtered)

b4, a4 = bandStopFilter(20, 60, sampling, order)
BS_Filtered = signal.filtfilt(b4, a4, data)
BS_FFT = fft(BS_Filtered)



plt.plot(t,data)
plt.title('ECG (Original)')
plt.savefig("ECGOriginal.png", dpi = 600)
plt.show()

plt.plot(t,LP_Filtered)
plt.title('ECG (Low Pass Filtered), Order: '+str(order))
plt.savefig("LowPassOrder"+str(order)+".png", dpi = 600)
plt.show()
plt.title('Comparison of Original and Filtered (LP) Signal')
plt.semilogy(x_FFT, 2.0/samples * np.abs(ECG_FFT[0:samples//2]), label = 'Original')
plt.semilogy(x_FFT, 2.0/samples * np.abs(LP_FFT[0:samples//2]), label = 'Filtered' , color ="yellowgreen")
plt.grid()
plt.legend()
plt.savefig("FFTLowOrder"+str(order)+".png", dpi = 600)
plt.show()

plt.plot(t,HP_Filtered)
plt.title('ECG (High Pass Filtered), Order: '+str(order))
plt.savefig("HighPassOrder"+str(order)+".png", dpi = 600)
plt.show()
plt.title('Comparison of Original and Filtered (HP) Signal')
plt.semilogy(x_FFT, 2.0/samples * np.abs(ECG_FFT[0:samples//2]), label = 'Original')
plt.semilogy(x_FFT, 2.0/samples * np.abs(HP_FFT[0:samples//2]),"yellowgreen",label = 'Filtered')
plt.grid()
plt.legend()
plt.savefig("FFTHighOrder"+str(order)+".png", dpi = 600)
plt.show()


plt.plot(t,BP_Filtered)
plt.title('ECG (Band Pass Filtered), Order: '+str(order))
plt.savefig("BandPassOrder"+str(order)+".png", dpi = 600)
plt.show()
plt.title('Comparison of Original and Filtered (BP) Signal')
plt.semilogy(x_FFT, 2.0/samples * np.abs(ECG_FFT[0:samples//2]), label = 'Original')
plt.semilogy(x_FFT, 2.0/samples * np.abs(BP_FFT[0:samples//2]),"yellowgreen", label = 'Filtered')
plt.grid()
plt.legend()
plt.savefig("FFTPassOrder"+str(order)+".png", dpi = 600)
plt.show()


plt.plot(t,BS_Filtered)
plt.title('ECG (Band Stop Filtered),  Order: '+str(order))
plt.savefig("BandStopOrder"+str(order)+".png", dpi = 600)
plt.show()
plt.title('Comparison of Original and Filtered (BS) Signal')
plt.semilogy(x_FFT, 2.0/samples * np.abs(ECG_FFT[0:samples//2]),label = 'Original')
plt.semilogy(x_FFT, 2.0/samples * np.abs(BS_FFT[0:samples//2]),"yellowgreen", label = 'Filtered')
plt.grid()
plt.legend()
plt.savefig("FFTStopOrder"+str(order)+".png", dpi = 600)
plt.show()
