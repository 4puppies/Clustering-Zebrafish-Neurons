import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


neuron_samples = np.genfromtxt(fname='sample_data.txt', delimiter=',')

plt.plot(range(0, 15000), neuron_samples[:, 0][0:15000])
plt.title('First 30 Seconds of Activity for a Neuron')
plt.xlabel('Time (s)')
plt.ylabel('Florescence Level')
plt.xlim([0, 15000])
plt.show()


sample_rate = 2 * len(neuron_samples)  # sample rate is twice the bandwidth
order = 4
cutoff = 20

b, a = butter_lowpass(cutoff, sample_rate, order)
w, h = freqz(b, a, worN=8000)

y1 = np.convolve(neuron_samples[:, 0], [1, -1])
y2 = np.array([50000 * butter_lowpass_filter(y1, cutoff, sample_rate, order)])  # there is a gain of 50000 here

neuron_signals = y2
graphing_signal = y2

for i in range(1, neuron_samples.shape[1]):
    y1 = np.convolve(neuron_samples[:, i], [1, -1])
    y2 = np.array([50000 * butter_lowpass_filter(y1, cutoff, sample_rate, order)])  # there is a gain of 50000 here

    neuron_signals = np.concatenate((neuron_signals, y2), axis=0)


plt.figure(1)
plt.subplot(211)
plt.title('Raw Signal')
plt.ylabel('Florescence Level')
plt.xlabel('Time (s)')
plt.plot(range(0, len(neuron_samples[:, 0])), neuron_samples[:, 0])

plt.subplot(212)
plt.plot(range(0, len(graphing_signal)), graphing_signal)
plt.title('Filtered Signal')
plt.ylabel('Florescence Level')
plt.xlabel('Time (s)')
plt.show()


n_clusters = 19
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(neuron_signals)
neuron_clusters = kmeans.labels_

print(neuron_clusters)

plt.hist(neuron_clusters, bins=range(0, n_clusters), alpha=0.5)
plt.xlabel('Cluster')
plt.ylabel('Number of Neurons')
plt.title('Number of Neurons that Fall into a Specific Cluster')
plt.show()

