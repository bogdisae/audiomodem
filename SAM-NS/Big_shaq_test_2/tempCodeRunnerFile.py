from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


fs_rx, rxSig = wavfile.read("SAM-NS/Big_shaq_test_2/Received_1.wav")
fs_tx, txChirp = wavfile.read("SAM-NS/chirp.wav")

rxSig = rxSig.astype(np.float32)
txChirp = txChirp.astype(np.float32)
rxSig /= np.max(np.abs(rxSig))
txChirp /= np.max(np.abs(txChirp))

corr = correlate(rxSig, rxSig, mode='full')
sync_index = np.argmax(np.abs(corr))
print("High correlation at sample:", sync_index)

# Sample indices for x-axis
x = np.arange(len(corr))

# Plot
plt.figure()
plt.plot(x, np.abs(corr))
plt.title("Matched Filter Output - Absolute value")
plt.xlabel("Sample index")
plt.ylabel("Correlation magnitude")
plt.show()

# Plot the original signal
x = np.arange(len(rxSig))
plt.figure()
plt.plot(x, rxSig)
plt.show()

# Plot the original signal
x = np.arange(len(txChirp))
plt.figure()
plt.plot(x, txChirp)
plt.show()
