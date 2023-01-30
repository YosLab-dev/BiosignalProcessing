import numpy as np
import scipy.io.wavfile as waves
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

#Reading the audio files
sampleRate1, audioA = waves.read("inglesEspaniol1.wav")
sampleRate2, audioB = waves.read("inglesEspaniol2.wav")
samples = len(audioA)
duration = samples/sampleRate1

t = np.arange(0, samples/sampleRate1, 1/sampleRate1)

audio = np.stack([audioA, audioB]).T
print(audio.shape)

ICA = FastICA(n_components=2)
ICA.fit(audio)
components = ICA.transform(audio)

waveform = np.transpose(components)
componentA = waveform[0]
componentB = waveform[1]

print(samples)
print(sampleRate1)
print(duration)

# Write the .wav file
waves.write("componentA.wav",sampleRate1,componentA)
waves.write("componentB.wav",sampleRate1,componentB)

#Plotting the audio signals
plt.plot(t, audioA)
plt.title('Audio A')
plt.savefig("Audio1.png", dpi = 600)
plt.show()

plt.plot(t, audioB)
plt.title('Audio B')
plt.savefig("Audio2.png", dpi = 600)
plt.show()

#Show the ICA components
plt.subplot(2,1,1)
plt.plot(t, componentA)
plt.ylabel('Component 1')

plt.subplot(2,1,2)
plt.plot(t, componentB)
plt.ylabel('Component 2')
plt.savefig("ICAComponents.png", dpi = 600)
plt.show()