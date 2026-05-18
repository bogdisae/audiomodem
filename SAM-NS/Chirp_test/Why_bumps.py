from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


fs_rx, rxChirp = wavfile.read("SAM-NS/Chirp_test/chirp.wav")
fs_tx, txChirp = wavfile.read("SAM-NS/Chirp_test/chirp.wav")

rxChirp = rxChirp.astype(np.float32)
txChirp = txChirp.astype(np.float32)

rxChirp /= np.max(np.abs(rxChirp))
txChirp /= np.max(np.abs(txChirp))


corr = correlate(rxChirp, txChirp, mode = "full")

# Sample indices for x-axis
x = np.arange(len(corr))

# Plot
plt.figure()
plt.plot(x, np.abs(corr))
plt.title("Matched Filter Output - Absolute value")
plt.xlabel("Sample index")
plt.ylabel("Correlation magnitude")
plt.show()
