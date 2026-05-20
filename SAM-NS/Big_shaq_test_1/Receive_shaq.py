from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


fs_rx, rxSig = wavfile.read("SAM-NS/big_shaq_received_2.wav")
fs_tx, txChirp = wavfile.read("SAM-NS/chirp.wav")

rxSig = rxSig.astype(np.float32)
txChirp = txChirp.astype(np.float32)

rxSig /= np.max(np.abs(rxSig))
txChirp /= np.max(np.abs(txChirp))

corr = correlate(rxSig, txChirp, mode='valid')
sync_index = np.argmax(np.abs(corr))
print("Chirp starts at sample:", sync_index)

# Sample indices for x-axis
x = np.arange(len(corr))

# Plot
plt.figure()
plt.plot(x, np.abs(corr))
plt.title("Matched Filter Output - Absolute value")
plt.xlabel("Sample index")
plt.ylabel("Correlation magnitude")
plt.show()

ofdm_start = sync_index + 2 * 96000 - 30 # Subtract 30 for safety

received_first_block = rxSig[ofdm_start: ofdm_start + 1024]

dft = np.fft.fft(received_first_block, 1024)
dft = dft[1:512]

